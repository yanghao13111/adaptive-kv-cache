"""
End-to-end benchmark runner.
Loads a config YAML, initializes the specified method, runs evaluation,
and writes results to experiments/results/.
"""

import argparse
import csv
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# suppress repetitive HuggingFace generation warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

from src.eval.metrics import compute_perplexity, measure_latency


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_texts(dataset_name: str, num_samples: int = 100) -> list[str]:
    """
    Load text samples from the specified dataset.
    Filters out empty or very short articles.
    """
    if dataset_name == "wikitext-103":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [x["text"] for x in dataset if len(x["text"].split()) > 100]
    elif dataset_name == "pg19":
        dataset = load_dataset("pg19", split="test")
        texts = [x["text"][:50000] for x in dataset if len(x["text"]) > 1000]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return texts[:num_samples]


def build_method(config: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """
    Return a callable fn(prompt) -> dict based on config method.
    """
    method = config["method"]
    kwargs = config.get("method_kwargs", {})
    max_new_tokens = config.get("max_new_tokens", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method == "full_cache":
        from src.baseline.full_cache import run_full_cache
        return lambda prompt: run_full_cache(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device)

    elif method == "sliding_window":
        from src.baseline.sliding_window import run_sliding_window
        return lambda prompt: run_sliding_window(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device, **kwargs)

    elif method == "naive_truncation":
        from src.baseline.naive_truncation import run_naive_truncation
        return lambda prompt: run_naive_truncation(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device, **kwargs)

    elif method == "adaptive":
        from src.adaptive.cache_manager import AdaptiveCacheManager
        from src.models.patched_attention import run_adaptive
        num_layers = model.config.num_hidden_layers
        cache_manager = AdaptiveCacheManager(num_layers=num_layers, **kwargs)
        return lambda prompt: run_adaptive(
            model, tokenizer, prompt, cache_manager,
            max_new_tokens=max_new_tokens, device=device,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def run_benchmark(config: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> dict:
    """
    Run full evaluation: latency + perplexity over dataset samples.
    Returns averaged metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config.get("dataset", "wikitext-103")
    num_samples = config.get("num_samples", 50)
    max_new_tokens = config.get("max_new_tokens", 512)

    print(f"Loading dataset: {dataset_name} ({num_samples} samples)")
    texts = load_texts(dataset_name, num_samples=num_samples)

    run_fn = build_method(config, model, tokenizer)

    # --- latency and memory: run generate on each text as prompt ---
    print("Running latency benchmark...")
    latencies, throughputs, memories = [], [], []
    for i, text in enumerate(texts):
        prompt = text[:500]  # use first 500 chars as prompt
        result = run_fn(prompt)
        latencies.append(result["latency_ms_per_token"])
        throughputs.append(result["throughput_tokens_per_sec"])
        memories.append(result["peak_memory_gb"])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(texts)}] avg latency so far: {sum(latencies)/len(latencies):.2f} ms/token")

    # --- perplexity: evaluate NLL on full texts ---
    print("Computing perplexity...")
    perplexity = compute_perplexity(
        model, tokenizer, texts,
        device=device,
        max_length=config.get("context_len", 4096),
    )

    return {
        "method": config["method"],
        "model": config["model"],
        "dataset": dataset_name,
        "num_samples": len(texts),
        "avg_latency_ms_per_token": round(sum(latencies) / len(latencies), 4),
        "avg_throughput_tokens_per_sec": round(sum(throughputs) / len(throughputs), 4),
        "avg_peak_memory_gb": round(sum(memories) / len(memories), 4),
        "perplexity": round(perplexity, 4),
    }


def compute_perplexity_with_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    cache_update_fn=None,
    cache_reset_fn=None,
    device: str = "cuda",
    max_length: int = 2048,
) -> float:
    """
    Compute perplexity autoregressively under a specific cache management strategy.

    Unlike compute_perplexity() which uses teacher-forcing (full context always
    visible), this function runs token-by-token decoding and applies cache_update_fn
    after each step. This means the model genuinely operates under the eviction or
    compression constraints, giving a realistic perplexity for that method.

    Args:
        texts:           Reference texts to evaluate on.
        cache_update_fn: Called after each decoding step as
                         fn(past_key_values, attentions) -> past_key_values.
                         Pass None for full-cache (no eviction).
                         Attentions are always captured and passed; the fn may
                         ignore them if not needed (e.g. sliding window).
        cache_reset_fn:  Called between texts to reset any stateful cache manager
                         (e.g. AdaptiveCacheManager.reset). Pass None if stateless.
        max_length:      Maximum number of tokens to evaluate per text.

    Returns:
        Average perplexity across all texts (lower is better).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if cache_reset_fn is not None:
                cache_reset_fn()

            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            input_ids = input_ids[:, :max_length]
            seq_len = input_ids.shape[1]
            if seq_len < 2:
                continue

            past_key_values = None
            # Feed first token without scoring, then score each subsequent token
            for pos in range(seq_len - 1):
                current_ids = input_ids[:, pos:pos + 1]
                outputs = model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=(cache_update_fn is not None),
                )
                past_key_values = outputs.past_key_values

                if cache_update_fn is not None:
                    past_key_values = cache_update_fn(past_key_values, outputs.attentions)

                # Score the next token
                log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
                next_token_id = input_ids[0, pos + 1].item()
                total_nll -= log_probs[0, next_token_id].item()
                total_tokens += 1

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated — check that texts are non-empty.")

    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


def build_cache_fn(config: dict, model: AutoModelForCausalLM):
    """
    Return (cache_update_fn, cache_reset_fn) for compute_perplexity_with_cache().

    cache_update_fn: fn(past_key_values, attentions) -> past_key_values, or None.
    cache_reset_fn:  fn() called between texts to clear stateful managers, or None.
    """
    method = config["method"]
    kwargs = config.get("method_kwargs", {})

    if method in ("full_cache", "naive_truncation"):
        return None, None

    elif method == "sliding_window":
        window_size = kwargs.get("window_size", 256)

        def sliding_fn(past_key_values, attentions):
            for layer in past_key_values.layers:
                if layer.keys.shape[2] > window_size:
                    layer.keys = layer.keys[:, :, -window_size:, :]
                    layer.values = layer.values[:, :, -window_size:, :]
            return past_key_values

        return sliding_fn, None

    elif method == "adaptive":
        from src.adaptive.cache_manager import AdaptiveCacheManager
        num_layers = model.config.num_hidden_layers
        cache_manager = AdaptiveCacheManager(num_layers=num_layers, **kwargs)
        return cache_manager.step, cache_manager.reset

    else:
        raise ValueError(f"Unknown method: {method}")


def run_benchmark_official(
    config: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Stage 4 benchmark: evaluates each method with its own cache management strategy.

    Perplexity is computed autoregressively via compute_perplexity_with_cache(),
    so eviction and compression genuinely affect the scores — unlike run_benchmark()
    which uses teacher-forcing and gives identical perplexity for all methods.

    Args:
        config: Same format as run_benchmark(). 'method' and 'method_kwargs' are used
                to build the appropriate cache_update_fn.

    Returns:
        Dict with latency, throughput, memory, and perplexity metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config.get("dataset", "wikitext-103")
    num_samples = config.get("num_samples", 50)
    max_length = config.get("context_len", 2048)

    print(f"Loading dataset: {dataset_name} ({num_samples} samples)")
    texts = load_texts(dataset_name, num_samples=num_samples)

    run_fn = build_method(config, model, tokenizer)

    # Latency and memory
    print("Running latency benchmark...")
    latencies, throughputs, memories = [], [], []
    for i, text in enumerate(texts):
        result = run_fn(text[:500])
        latencies.append(result["latency_ms_per_token"])
        throughputs.append(result["throughput_tokens_per_sec"])
        memories.append(result["peak_memory_gb"])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(texts)}] avg latency: {sum(latencies)/len(latencies):.2f} ms/token")

    # Perplexity under this method's cache strategy
    print("Computing perplexity (autoregressive)...")
    cache_fn, reset_fn = build_cache_fn(config, model)

    # Naive truncation: evaluate only on the first max_cache_size tokens
    eval_max_length = config.get("method_kwargs", {}).get("max_cache_size", max_length) \
        if config["method"] == "naive_truncation" else max_length

    perplexity = compute_perplexity_with_cache(
        model, tokenizer, texts,
        cache_update_fn=cache_fn,
        cache_reset_fn=reset_fn,
        device=device,
        max_length=eval_max_length,
    )

    return {
        "method": config["method"],
        "model": config["model"],
        "dataset": dataset_name,
        "num_samples": len(texts),
        "avg_latency_ms_per_token": round(sum(latencies) / len(latencies), 4),
        "avg_throughput_tokens_per_sec": round(sum(throughputs) / len(throughputs), 4),
        "avg_peak_memory_gb": round(sum(memories) / len(memories), 4),
        "perplexity": round(perplexity, 4),
    }


def save_results(results: dict, output_dir: str = "experiments/results") -> str:
    """Save results dict to a timestamped CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{results['method']}_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to {filename}")
    return str(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model in config")
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--memory_budget_gb", type=float, default=None)
    parser.add_argument("--recent_window", type=int, default=None)
    parser.add_argument("--compress_dtype", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["model"] = args.model
    if args.context_len:
        config["context_len"] = args.context_len
    if args.num_samples:
        config["num_samples"] = args.num_samples
    if args.memory_budget_gb:
        config.setdefault("method_kwargs", {})["memory_budget_gb"] = args.memory_budget_gb
    if args.recent_window:
        config.setdefault("method_kwargs", {})["recent_window"] = args.recent_window
    if args.compress_dtype:
        config.setdefault("method_kwargs", {})["compress_dtype"] = args.compress_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {config['model']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    results = run_benchmark(config, model, tokenizer)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    save_results(results)


if __name__ == "__main__":
    main()

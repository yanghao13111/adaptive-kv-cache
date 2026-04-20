"""
End-to-end benchmark runner.
Loads a config YAML, initializes the specified method, runs evaluation,
and writes results to experiments/results/.
"""

import argparse
import csv
import yaml
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        from src.models.patched_attention import patch_model
        cache_manager = AdaptiveCacheManager(**kwargs)
        patch_model(model, cache_manager)
        raise NotImplementedError("Adaptive decoding runner not yet implemented")

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

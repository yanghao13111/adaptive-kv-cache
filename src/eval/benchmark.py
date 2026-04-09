"""
End-to-end benchmark runner.
Loads a config YAML, initializes the specified method, runs evaluation,
and writes results to experiments/results/.
"""

import argparse
import yaml
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import compute_perplexity, measure_latency


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_method(config: dict, model: AutoModelForCausalLM):
    """
    Instantiate the decoding method specified in the config.
    Returns a callable: fn(prompt) -> dict.
    """
    method = config["method"]

    if method == "full_cache":
        from src.baseline.full_cache import run_full_cache
        return lambda prompt: run_full_cache(model, tokenizer, prompt, **config.get("method_kwargs", {}))

    elif method == "sliding_window":
        from src.baseline.sliding_window import run_sliding_window
        return lambda prompt: run_sliding_window(model, tokenizer, prompt, **config.get("method_kwargs", {}))

    elif method == "naive_truncation":
        from src.baseline.naive_truncation import run_naive_truncation
        return lambda prompt: run_naive_truncation(model, tokenizer, prompt, **config.get("method_kwargs", {}))

    elif method == "adaptive":
        from src.adaptive.cache_manager import AdaptiveCacheManager
        from src.models.patched_attention import patch_model
        cache_manager = AdaptiveCacheManager(**config.get("method_kwargs", {}))
        patch_model(model, cache_manager)
        # TODO: return adaptive decoding callable
        raise NotImplementedError("Adaptive decoding runner not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model in config")
    parser.add_argument("--context_len", type=int, default=None)
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
    if args.memory_budget_gb:
        config.setdefault("method_kwargs", {})["memory_budget_gb"] = args.memory_budget_gb
    if args.recent_window:
        config.setdefault("method_kwargs", {})["recent_window"] = args.recent_window
    if args.compress_dtype:
        config.setdefault("method_kwargs", {})["compress_dtype"] = args.compress_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {config['model']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    run_fn = build_method(config, model)

    # TODO: load dataset, run evaluation, collect metrics, save results
    raise NotImplementedError("Dataset loading and result aggregation not yet implemented")


if __name__ == "__main__":
    main()

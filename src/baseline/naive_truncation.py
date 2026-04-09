"""
Naive cache truncation baseline.
Hard-cuts the KV-cache to a fixed maximum size, dropping the oldest tokens.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_naive_truncation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_cache_size: int = 1024,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Run autoregressive decoding with hard cache truncation.
    When cache exceeds max_cache_size, oldest tokens are dropped.

    Returns:
        dict with keys: generated_text, latency_ms_per_token, peak_memory_gb
    """
    # TODO: implement
    raise NotImplementedError

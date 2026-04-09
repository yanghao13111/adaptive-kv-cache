"""
Fixed sliding-window KV-cache eviction baseline.
Only the most recent W tokens are retained in the cache.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_sliding_window(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    window_size: int = 512,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Run autoregressive decoding with a fixed sliding-window KV-cache.
    Tokens outside the window are dropped unconditionally.

    Returns:
        dict with keys: generated_text, latency_ms_per_token, peak_memory_gb
    """
    # TODO: implement
    raise NotImplementedError

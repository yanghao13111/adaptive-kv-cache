"""
Standard full KV-cache decoding baseline.
No eviction or compression; cache grows unbounded.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_full_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Run autoregressive decoding with full KV-cache (HuggingFace default).

    Returns:
        dict with keys: generated_text, latency_ms_per_token, peak_memory_gb
    """
    # TODO: implement
    raise NotImplementedError

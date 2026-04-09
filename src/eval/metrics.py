"""
Evaluation metrics: perplexity, throughput, and GPU memory tracking.
"""

import time
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 4096,
) -> float:
    """
    Compute average perplexity over a list of text samples.

    Args:
        texts: List of reference texts (e.g. from WikiText-103 or PG-19).

    Returns:
        Average perplexity (lower is better).
    """
    # TODO: implement sliding-window perplexity for long texts
    raise NotImplementedError


def measure_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "cuda",
    warmup_steps: int = 2,
) -> dict:
    """
    Measure per-token decoding latency and throughput.

    Returns:
        dict with keys:
            latency_ms_per_token  (float)
            throughput_tokens_per_sec  (float)
            peak_memory_gb  (float)
    """
    # TODO: implement
    raise NotImplementedError


def get_peak_memory_gb() -> float:
    """Return peak GPU memory allocated since last reset, in GB."""
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_peak_memory() -> None:
    """Reset the peak GPU memory counter."""
    torch.cuda.reset_peak_memory_stats()

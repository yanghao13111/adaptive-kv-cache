"""
Evaluation metrics: perplexity, throughput, and GPU memory tracking.
"""

import time
import torch
import numpy as np
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 4096,
    stride: int = 512,
) -> float:
    """
    Compute average perplexity over a list of text samples.

    Uses a sliding window to handle texts longer than max_length,
    so we evaluate on the full document rather than truncating.

    Args:
        texts: List of reference texts (e.g. from WikiText-103 or PG-19).
        stride: Step size for sliding window. Smaller = more accurate but slower.

    Returns:
        Average perplexity (lower is better).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            prev_end = 0
            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                # tokens we actually score this window (exclude overlap with previous window)
                target_len = end - max(begin, prev_end)
                if target_len <= 0:
                    prev_end = end
                    continue

                window_ids = input_ids[:, begin:end]
                labels = window_ids.clone()
                # mask out the overlap region — only score new tokens
                labels[:, :-target_len] = -100

                outputs = model(window_ids, labels=labels)
                # outputs.loss is mean NLL over non-masked tokens
                total_nll += outputs.loss.item() * target_len
                total_tokens += target_len
                prev_end = end

                if end == seq_len:
                    break

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated — check that texts are non-empty.")

    avg_nll = total_nll / total_tokens
    return float(np.exp(avg_nll))


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

    Runs warmup iterations first to stabilize GPU state before timing.

    Returns:
        dict with keys:
            latency_ms_per_token  (float)
            throughput_tokens_per_sec  (float)
            peak_memory_gb  (float)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # warmup — GPU has initialization overhead on first run
    with torch.no_grad():
        for _ in range(warmup_steps):
            model.generate(**inputs, max_new_tokens=max_new_tokens)

    reset_peak_memory()

    if device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    # count only the newly generated tokens, not the prompt
    n_new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    elapsed_sec = end - start

    latency_ms = (elapsed_sec / n_new_tokens) * 1000
    throughput = n_new_tokens / elapsed_sec
    peak_mem = get_peak_memory_gb()

    return {
        "latency_ms_per_token": latency_ms,
        "throughput_tokens_per_sec": throughput,
        "peak_memory_gb": peak_mem,
    }


def get_peak_memory_gb() -> float:
    """Return peak GPU memory allocated since last reset, in GB."""
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_peak_memory() -> None:
    """Reset the peak GPU memory counter."""
    torch.cuda.reset_peak_memory_stats()

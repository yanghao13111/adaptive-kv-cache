"""
Per-step KV-cache latency profiler.
"""

import time
import torch
from typing import Literal


def measure_per_step_latency(
    method: Literal["full_cache", "sliding_window", "adaptive"],
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    device: str = "cuda",
    window_size: int = 256,
    cache_manager=None,
) -> list[dict]:
    """
    Run a manual decode loop and record per-step latency and cache size.

    Prefill (processing all input_ids) is run once before the timed loop so
    that every measured step processes exactly one new token.

    Args:
        method:         "full_cache", "sliding_window", or "adaptive".
        model:          HuggingFace causal LM (on device, eval mode).
        input_ids:      Tokenized prompt, shape (1, seq_len), on device.
        max_new_tokens: Number of decode steps to time.
        device:         "cuda" or "cpu".
        window_size:    Tokens to keep (sliding_window only).
        cache_manager:  AdaptiveCacheManager instance (adaptive only).
                        Call cache_manager.reset() before each new sample.

    Returns:
        List of dicts: [{"step": int, "cache_tokens": int, "latency_ms": float}, ...]
        step 0 = first decode step after prefill.
    """
    if method == "adaptive":
        assert cache_manager is not None, "cache_manager required for adaptive"

    use_cuda = device == "cuda"
    use_attentions = method == "adaptive"

    model.eval()
    records = []

    with torch.no_grad():
        # --- Prefill (not timed) ---
        prefill_outputs = model(
            input_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=use_attentions,
        )
        past_key_values = prefill_outputs.past_key_values

        if method == "sliding_window":
            cache_len = past_key_values.layers[0].keys.shape[2]
            if cache_len > window_size:
                for layer in past_key_values.layers:
                    layer.keys = layer.keys[:, :, -window_size:, :]
                    layer.values = layer.values[:, :, -window_size:, :]
        elif method == "adaptive":
            past_key_values = cache_manager.step(past_key_values, prefill_outputs.attentions)

        current_input = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- Timed decode loop (1 token per step) ---
        for step in range(max_new_tokens):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=use_attentions,
            )

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            past_key_values = outputs.past_key_values

            if method == "sliding_window":
                cache_len = past_key_values.layers[0].keys.shape[2]
                if cache_len > window_size:
                    for layer in past_key_values.layers:
                        layer.keys = layer.keys[:, :, -window_size:, :]
                        layer.values = layer.values[:, :, -window_size:, :]
            elif method == "adaptive":
                past_key_values = cache_manager.step(past_key_values, outputs.attentions)

            cache_tokens = past_key_values.layers[0].keys.shape[2]
            records.append({
                "step": step,
                "cache_tokens": cache_tokens,
                "latency_ms": (t1 - t0) * 1000,
            })

            current_input = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return records

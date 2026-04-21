"""
Fixed sliding-window KV-cache eviction baseline.
Only the most recent W tokens are retained in the cache.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import reset_peak_memory, get_peak_memory_gb


def run_sliding_window(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    window_size: int = 512,
    max_new_tokens: int = 512,
    device: str = "cuda",
    warmup_steps: int = 2,
) -> dict:
    """
    Run autoregressive decoding with a fixed sliding-window KV-cache.
    Tokens outside the window are evicted at every decoding step.

    Returns:
        dict with keys: generated_text, latency_ms_per_token,
                        throughput_tokens_per_sec, peak_memory_gb
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    input_ids = inputs["input_ids"]

    def _decode(track_memory: bool):
        past_key_values = None
        current_input = input_ids
        generated_ids = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

                # Evict oldest tokens if KV cache exceeds window_size.
                # DynamicCache stores keys/values per layer; we slice the seq dimension to keep last window_size tokens.
                # crop(N) keeps the first N tokens so we cannot use it for sliding window (we want the last N).
                cache_len = past_key_values.layers[0].keys.shape[2]
                if cache_len > window_size:
                    for layer in past_key_values.layers:
                        layer.keys = layer.keys[:, :, -window_size:, :]
                        layer.values = layer.values[:, :, -window_size:, :]

                # Greedy decoding — take the most probable next token
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                current_input = next_token

                # Stop if EOS token is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break

        return generated_ids

    # Warmup to stabilize GPU state before timing
    for _ in range(warmup_steps):
        _decode(track_memory=False)

    reset_peak_memory()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    generated_ids = _decode(track_memory=True)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    n_tokens = len(generated_ids)
    elapsed_sec = end - start

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "latency_ms_per_token": (elapsed_sec / n_tokens) * 1000 if n_tokens > 0 else 0.0,
        "throughput_tokens_per_sec": n_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
        "peak_memory_gb": get_peak_memory_gb(),
    }

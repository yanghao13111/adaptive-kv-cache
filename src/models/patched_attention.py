"""
Adaptive KV-cache decoding loop.
Uses output_attentions=True to capture attention weights at each step,
then passes them to AdaptiveCacheManager for tier management.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.adaptive.cache_manager import AdaptiveCacheManager
from src.eval.metrics import reset_peak_memory, get_peak_memory_gb


def run_adaptive(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    cache_manager: AdaptiveCacheManager,
    max_new_tokens: int = 512,
    device: str = "cuda",
    warmup_steps: int = 2,
) -> dict:
    """
    Run autoregressive decoding with adaptive KV-cache management.

    At each step:
      1. Run forward pass with output_attentions=True to get attention weights.
      2. Pass attention weights to cache_manager.step() for tier management.
      3. Greedy decode the next token.

    Args:
        model:         HuggingFace causal LM.
        tokenizer:     Corresponding tokenizer.
        prompt:        Input text.
        cache_manager: AdaptiveCacheManager instance (pre-configured).
        max_new_tokens: Maximum number of tokens to generate.
        device:        "cuda" or "cpu".
        warmup_steps:  Number of warmup runs before timing.

    Returns:
        dict with keys: generated_text, latency_ms_per_token,
                        throughput_tokens_per_sec, peak_memory_gb
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    input_ids = inputs["input_ids"]

    def _decode():
        cache_manager.reset()
        past_key_values = None
        current_input = input_ids
        generated_ids = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )

                past_key_values = cache_manager.step(
                    outputs.past_key_values,
                    outputs.attentions,
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                current_input = next_token

                if next_token.item() == tokenizer.eos_token_id:
                    break

        return generated_ids

    # Warmup
    for _ in range(warmup_steps):
        _decode()

    reset_peak_memory()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    generated_ids = _decode()
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

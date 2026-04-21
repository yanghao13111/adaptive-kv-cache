"""
Naive cache truncation baseline.
Truncates the prompt to max_cache_size tokens before generation.
KV cache grows unbounded during generation — only the input is truncated.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import measure_latency


def run_naive_truncation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_cache_size: int = 1024,
    max_new_tokens: int = 512,
    device: str = "cuda",
    warmup_steps: int = 2,
) -> dict:
    """
    Run autoregressive decoding with hard prompt truncation.
    If the prompt exceeds max_cache_size tokens, the oldest tokens are dropped
    before generation begins. KV cache is not managed during generation.

    Returns:
        dict with keys: generated_text, latency_ms_per_token,
                        throughput_tokens_per_sec, peak_memory_gb
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)

    # Truncate prompt to max_cache_size tokens if needed
    if inputs["input_ids"].shape[1] > max_cache_size:
        inputs["input_ids"] = inputs["input_ids"][:, -max_cache_size:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_cache_size:]

    truncated_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    metrics = measure_latency(
        model, tokenizer, truncated_prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        warmup_steps=warmup_steps,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return {
        "generated_text": generated_text,
        **metrics,
    }

"""
Standard full KV-cache decoding baseline.
No eviction or compression; cache grows unbounded.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import measure_latency


def run_full_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
    warmup_steps: int = 2,
) -> dict:
    """
    Run autoregressive decoding with full KV-cache (HuggingFace default).

    Returns:
        dict with keys: generated_text, latency_ms_per_token,
                        throughput_tokens_per_sec, peak_memory_gb
    """
    metrics = measure_latency(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        warmup_steps=warmup_steps,
    )

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
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

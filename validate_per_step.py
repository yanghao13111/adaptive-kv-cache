"""
Sanity check for measure_per_step_latency using gpt2 on CPU.
Verifies cache behavior (not latency accuracy) for all three methods.

Run from project root:
    python validate_per_step.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.per_step import measure_per_step_latency
from src.adaptive.cache_manager import AdaptiveCacheManager

MODEL = "gpt2"
DEVICE = "cpu"
MAX_NEW_TOKENS = 30
WINDOW_SIZE = 20
# gpt2: 12 layers, 12 heads, head_dim=64, FP16 calc → ~0.0000343 GB/token
# budget=0.001 GB → eviction triggers at ~29 tokens; prefill=30 → triggers on step 1
BUDGET_GB = 0.001

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

text = "The history of artificial intelligence spans decades of research and development. " * 3
input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :30].to(DEVICE)
prefill_len = input_ids.shape[1]
print(f"Prefill: {prefill_len} tokens | Decode: {MAX_NEW_TOKENS} steps")
print(f"sliding_window size: {WINDOW_SIZE} | adaptive budget: {BUDGET_GB} GB\n")

results = {}

print("Running full_cache...")
results["full_cache"] = measure_per_step_latency(
    "full_cache", model, input_ids, MAX_NEW_TOKENS, DEVICE
)

print("Running sliding_window...")
results["sliding_window"] = measure_per_step_latency(
    "sliding_window", model, input_ids, MAX_NEW_TOKENS, DEVICE,
    window_size=WINDOW_SIZE,
)

print("Running adaptive...")
cache_manager = AdaptiveCacheManager(
    num_layers=model.config.n_layer,
    memory_budget_gb=BUDGET_GB,
    recent_window=10,
    sink_tokens=2,
)
results["adaptive"] = measure_per_step_latency(
    "adaptive", model, input_ids, MAX_NEW_TOKENS, DEVICE,
    cache_manager=cache_manager,
)

# Print table
print(f"\n{'Step':>4}  {'full_cache':>10}  {'sliding_win':>11}  {'adaptive':>8}")
print("-" * 42)
for step in range(MAX_NEW_TOKENS):
    fc = results["full_cache"][step]["cache_tokens"]
    sw = results["sliding_window"][step]["cache_tokens"]
    ad = results["adaptive"][step]["cache_tokens"]
    print(f"{step:>4}  {fc:>10}  {sw:>11}  {ad:>8}")

# Assertions
print("\n--- Validation ---")
passes = 0

fc_tokens = [r["cache_tokens"] for r in results["full_cache"]]
if all(fc_tokens[i] > fc_tokens[i - 1] for i in range(1, len(fc_tokens))):
    print("PASS  full_cache: cache grows every step")
    passes += 1
else:
    print("FAIL  full_cache: cache did not grow monotonically")
    print(f"      tokens: {fc_tokens}")

sw_tokens = [r["cache_tokens"] for r in results["sliding_window"]]
if all(t <= WINDOW_SIZE for t in sw_tokens):
    print(f"PASS  sliding_window: cache bounded at {WINDOW_SIZE}")
    passes += 1
else:
    print(f"FAIL  sliding_window: some steps exceeded window_size={WINDOW_SIZE}")
    print(f"      tokens: {sw_tokens}")

ad_tokens = [r["cache_tokens"] for r in results["adaptive"]]
max_unbounded = prefill_len + MAX_NEW_TOKENS
if max(ad_tokens) < max_unbounded:
    print(f"PASS  adaptive: eviction triggered (max cache={max(ad_tokens)} < {max_unbounded})")
    passes += 1
else:
    print(f"FAIL  adaptive: eviction did not trigger (max={max(ad_tokens)}, expected < {max_unbounded})")

print(f"\n{passes}/3 passed")

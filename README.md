# Adaptive KV Cache Compression and Eviction for Efficient LLM Decoding

## Overview

Large language model (LLM) inference is increasingly bottlenecked by the memory and latency overhead of the key-value (KV) cache during autoregressive decoding, especially for long-context generation.

This project implements an **adaptive KV cache optimization framework** that reduces peak GPU memory usage while preserving generation quality. We partition cached tokens into three tiers based on recency and estimated importance:

| Tier | Tokens | Policy |
|------|--------|--------|
| **Recent** | Last *W* tokens | Kept in full precision (FP16/BF16) |
| **Moderate** | Historically attended tokens | Compressed to low precision (INT4/INT8) |
| **Old / Low-importance** | Low attention-score tokens | Evicted when memory budget exceeded |

We evaluate the quality–efficiency tradeoff across peak GPU memory, decoding latency, throughput (tokens/sec), and output quality (perplexity) on long-context generation tasks.

---

## Repository Structure

```
adaptive-kv-cache/
├── README.md
├── requirements.txt
│
├── src/
│   ├── baseline/
│   │   ├── full_cache.py          # Standard full KV-cache decoding
│   │   ├── sliding_window.py      # Fixed sliding-window eviction
│   │   └── naive_truncation.py    # Naive cache truncation
│   │
│   ├── adaptive/
│   │   ├── cache_manager.py       # Core: tier partitioning + budget logic
│   │   ├── importance_scorer.py   # Attention-based importance heuristics
│   │   ├── compressor.py          # Low-precision quantization (INT4/INT8)
│   │   └── eviction_policy.py     # Budget-triggered eviction logic
│   │
│   ├── models/
│   │   └── patched_attention.py   # HuggingFace attention forward-pass hook
│   │
│   └── eval/
│       ├── metrics.py             # Perplexity, throughput, memory tracking
│       └── benchmark.py           # End-to-end benchmark runner
│
├── experiments/
│   ├── configs/
│   │   ├── baseline_full.yaml
│   │   ├── baseline_sliding.yaml
│   │   └── adaptive_main.yaml
│   └── results/                   # Output CSVs and plots (gitignored raw data)
│
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_adaptive_ablation.ipynb
│   └── 03_results_visualization.ipynb
│
└── report/
    └── final_report.pdf           # (added at submission)
```

---

## Method

### Three-Tier Token Partitioning

At each decoding step, all cached tokens are classified into one of three regions:

```
[  old tokens  |  moderate tokens  |  recent tokens  ] → [new token]
     evict           compress            full precision
```

**Importance scoring** uses an exponentially decayed attention heuristic: each step, existing scores are multiplied by a decay factor (default 0.9) before adding the new step's attention weights. This prevents early tokens from accumulating inflated scores simply by being present longer, ensuring eviction targets tokens that are *currently* unimportant.

**Protected zones** — two token groups are never evicted or compressed:
- **Attention sinks** (first 4 tokens): the model routes disproportionate attention to initial tokens regardless of content ([StreamingLLM](https://arxiv.org/abs/2309.17453)). Evicting them causes perplexity spikes on long sequences.
- **Recent window** (last 256 tokens): newly generated tokens have not yet accumulated enough attention history to score fairly.

**Memory budget** is set as a fixed GPU memory cap (e.g., 4 GB for the KV cache). When the cache exceeds this budget, the lowest-scored tokens in the evictable zone are removed first.

### Baselines

1. **Full KV cache** — standard HuggingFace decoding, no eviction
2. **Sliding window** — keep only the most recent *N* tokens
3. **Naive truncation** — hard-cut cache at a fixed size

---

## Project Roadmap

- [x] **Stage 1** — Baseline inference pipeline and full KV-cache benchmarking
- [x] **Stage 2** — Simple baselines: sliding-window eviction and naive truncation
- [x] **Stage 3** — Adaptive method: recency-aware retention, compression, budget-triggered eviction
- [ ] **Stage 4** — Official experiments on Mistral-7B: WikiText-103 (2048 tokens) + LongBench qasper (4096 tokens)
- [ ] **Stage 5** — Ablation sweep: memory budget, recent window size, compress dtype
- [ ] **Stage 6** — Results visualization and writeup

---

## Confirmed Setup

| Component | Choice | Reason |
|-----------|--------|--------|
| Validation model | `gpt2` | Fast, no GPU needed, quick iteration |
| Official experiment model | `mistralai/Mistral-7B-v0.1` | No auth required, commonly used in KV cache papers, GQA architecture |
| Primary dataset | LongBench (qasper) | Long-context evaluation (avg 3600 words); only dataset where KV cache pressure is high enough to trigger eviction and show memory savings |
| Secondary dataset | WikiText-103 | Verifies quality is preserved when eviction does not trigger (sequences too short to exceed budget) |
| Experiment environment | Google Colab Pro (A100) | Full CUDA support, bitsandbytes compatible |

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, HuggingFace Transformers, bitsandbytes (for INT4/INT8 quantization)

### Tested environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.2.0 |
| Transformers | 4.40.x |
| CUDA | 11.8 |

---

## Running Experiments

### 1. Benchmark the full-cache baseline

```bash
python src/eval/benchmark.py \
  --config experiments/configs/baseline_full.yaml \
  --model meta-llama/Llama-2-7b-hf \
  --context_len 4096
```

### 2. Run the adaptive method

```bash
python src/eval/benchmark.py \
  --config experiments/configs/adaptive_main.yaml \
  --model meta-llama/Llama-2-7b-hf \
  --context_len 4096 \
  --memory_budget_gb 4.0 \
  --recent_window 256 \
  --compress_dtype int8
```

---

## Evaluation Metrics

| Metric | Tool / Method |
|--------|--------------|
| Peak GPU memory | `torch.cuda.max_memory_allocated()` |
| Decoding latency (ms/token) | Wall-clock time per generated token |
| Throughput (tokens/sec) | Batch-averaged token generation rate |
| Output quality | Perplexity on WikiText-103 / PG-19 |

---

## Results

> *(To be filled in after experiments are complete)*

**WikiText-103** (context_len=2048)

| Method | Peak Memory (GB) | Latency (ms/tok) | Throughput (tok/s) | Perplexity |
|--------|-----------------|------------------|--------------------|------------|
| Full KV cache | — | — | — | — |
| Sliding window | — | — | — | — |
| Naive truncation | — | — | — | — |
| **Adaptive (ours)** | — | — | — | — |

**LongBench / qasper** (context_len=4096)

| Method | Peak Memory (GB) | Latency (ms/tok) | Throughput (tok/s) | Perplexity |
|--------|-----------------|------------------|--------------------|------------|
| Full KV cache | — | — | — | — |
| Sliding window | — | — | — | — |
| Naive truncation | — | — | — | — |
| **Adaptive (ours)** | — | — | — | — |

---


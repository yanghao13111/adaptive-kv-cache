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
│   └── results/
│       ├── combined_results.csv       # Final benchmark results (all methods × datasets)
│       └── figures/                   # Generated plots (300 DPI PNG)
│
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_adaptive_validation.ipynb
│   ├── 03_official_experiments.ipynb
│   └── 04_visualization.ipynb         # Stage 5: result figures
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

**Memory budget** is set as a fixed GPU memory cap (0.2 GB for the KV cache in our experiments). When the cache exceeds this budget, the lowest-scored tokens in the evictable zone are removed first.

### Baselines

1. **Full KV cache** — standard HuggingFace decoding, no eviction
2. **Sliding window** — keep only the most recent *N* tokens
3. **Naive truncation** — hard-cut cache at a fixed size

---

## Project Roadmap

- [x] **Stage 1** — Baseline inference pipeline and full KV-cache benchmarking
- [x] **Stage 2** — Simple baselines: sliding-window eviction and naive truncation
- [x] **Stage 3** — Adaptive method: recency-aware retention, compression, budget-triggered eviction
- [x] **Stage 4** — Official experiments on Mistral-7B: WikiText-103 (2048 tokens) + LongBench qasper (4096 tokens)
- [x] **Stage 5** — Results visualization and writeup

---

## Confirmed Setup

| Component | Choice | Reason |
|-----------|--------|--------|
| Validation model | `gpt2` | Fast, no GPU needed, quick iteration |
| Official experiment model | `mistralai/Mistral-7B-v0.1` | No auth required, commonly used in KV cache papers, GQA architecture |
| Primary dataset | LongBench (qasper) | Long-context evaluation (avg 3600 words); only dataset where KV cache pressure is high enough to trigger eviction and show memory savings |
| Secondary dataset | WikiText-103 | Verifies quality is preserved when eviction does not trigger (sequences too short to exceed budget) |
| Experiment environment | Kaggle (T4 x2) | Full CUDA support, bitsandbytes compatible |

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
  --model mistralai/Mistral-7B-v0.1 \
  --context_len 4096
```

### 2. Run the adaptive method

```bash
python src/eval/benchmark.py \
  --config experiments/configs/adaptive_main.yaml \
  --model mistralai/Mistral-7B-v0.1 \
  --context_len 4096 \
  --memory_budget_gb 0.2 \
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
| Output quality | Perplexity on WikiText-103 and LongBench (qasper) |

---

## Results

Experiments run on Kaggle (T4 x2), model: `mistralai/Mistral-7B-v0.1`, memory budget: 0.2 GB.

> **Note:** Peak GPU memory includes model weights (~6.78 GB); the key efficiency metric is **KV Cache (GB)** which reflects only the cache tensor size.

**WikiText-103** (50 samples; sequences too short to exceed budget — eviction does not trigger)

| Method | KV Cache (GB) | Peak Memory (GB) | Latency (ms/tok) | Throughput (tok/s) | Perplexity |
|--------|--------------|-----------------|------------------|--------------------|------------|
| Full KV cache | 0.0280 | 6.7822 | 65.98 | 15.16 | 8.3445 |
| Sliding window | 0.0253 | 6.7770 | 65.29 | 15.32 | 9.5790 |
| Naive truncation | 0.0253 | 6.7822 | 65.95 | 15.16 | 8.8986 |
| **Adaptive (ours)** | **0.0280** | 6.8068 | 73.92 | 13.54 | **8.3446** |

**LongBench / qasper** (20 samples; long-context — eviction triggers, memory savings visible)

| Method | KV Cache (GB) | Peak Memory (GB) | Latency (ms/tok) | Throughput (tok/s) | Perplexity |
|--------|--------------|-----------------|------------------|--------------------|------------|
| Full KV cache | 0.4821 | 6.7805 | 67.28 | 14.93 | **5.7405** |
| Sliding window | 0.0312 | 6.7730 | 66.47 | 15.11 | 24.2441 |
| Naive truncation | 0.0311 | 6.7805 | 67.16 | 14.97 | 9.3677 |
| **Adaptive (ours)** | **0.2000** | 6.7907 | 73.50 | 13.65 | 5.8455 |

**Key takeaway:** On LongBench, adaptive compression reduces KV cache from 0.48 GB to 0.20 GB (**−58.5%**) with only +1.8% perplexity degradation (+0.105), while alternative methods that achieve similar memory savings (sliding window, naive truncation) suffer severe quality loss (perplexity 9.4–24.2 vs 5.74).

---


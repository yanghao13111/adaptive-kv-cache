"""
Core adaptive KV-cache manager.
Orchestrates three-tier token partitioning: full-precision (recent),
compressed (moderate), and evicted (low-importance).
"""

import torch
from torch import Tensor
from .importance_scorer import ImportanceScorer
from .compressor import CacheCompressor, DType
from .eviction_policy import EvictionPolicy


class AdaptiveCacheManager:
    """
    Manages the KV-cache during autoregressive decoding by partitioning
    tokens into three tiers and enforcing a memory budget.

    Tiers
    -----
    - Recent   : last `recent_window` tokens, kept in full precision.
    - Moderate : historically attended tokens, compressed to `compress_dtype`.
    - Evicted  : low-importance tokens removed from cache entirely.

    Token state machine (one direction only, no double-compression):
        full_precision → compressed → evicted
    """

    def __init__(
        self,
        memory_budget_gb: float = 4.0,
        recent_window: int = 256,
        compress_dtype: DType = "int8",
        num_layers: int = 12,
        compress_ratio: float = 0.5,
        sink_tokens: int = 4,
        score_decay: float = 0.9,
    ):
        """
        Args:
            memory_budget_gb: Maximum KV cache memory in GB before eviction triggers.
            recent_window:    Number of most-recent tokens always kept in full precision.
            compress_dtype:   Quantization dtype for moderate-tier tokens.
            num_layers:       Number of transformer layers (for ImportanceScorer).
            compress_ratio:   Fraction of non-recent tokens to compress (0.5 = bottom 50% by score get compressed).
            sink_tokens:      Number of initial prompt tokens protected as attention sinks.
            score_decay:      Exponential decay factor applied to scores each step (0.9 default).
        """
        self.recent_window = recent_window
        self.compress_ratio = compress_ratio
        self.sink_tokens = sink_tokens

        self.scorer = ImportanceScorer(num_layers=num_layers, score_decay=score_decay)
        self.compressor = CacheCompressor(dtype=compress_dtype)
        self.eviction_policy = EvictionPolicy(
            memory_budget_gb=memory_budget_gb,
            recent_window=recent_window,
            sink_tokens=sink_tokens,
        )

        # Track compression state per token: index → metadata (None = full precision)
        self._compressed: dict[int, dict] = {}

    def step(
        self,
        past_key_values,
        attentions: tuple[Tensor, ...],
    ):
        """
        Called after each decoding step to update scores and apply tier management.

        Args:
            past_key_values: HuggingFace DynamicCache object.
            attentions:      Tuple of per-layer attention tensors,
                             each shape (batch, heads, query_len, key_len).

        Returns:
            Updated past_key_values after eviction (compression tracked internally).
        """
        # 1. Update importance scores
        self.scorer.update(attentions)

        # Use layers[0].keys.shape[2] as the authoritative seq_len — reflects prior evictions
        seq_len = past_key_values.layers[0].keys.shape[2]
        scores = self.scorer.score(seq_len)

        # 2. Estimate current cache memory (keys + values, all layers, FP16 = 2 bytes)
        n_layers = len(past_key_values.layers)
        head_dim = past_key_values.layers[0].keys.shape[-1]
        n_heads = past_key_values.layers[0].keys.shape[1]
        bytes_per_token = n_layers * n_heads * head_dim * 2 * 2  # keys + values, FP16

        current_bytes = seq_len * bytes_per_token

        # 3. Evict if over memory budget
        if self.eviction_policy.should_evict(current_bytes):
            n_evict = self.eviction_policy.n_tokens_to_evict(current_bytes, bytes_per_token)
            evict_indices = self.eviction_policy.select_eviction_indices(scores, n_evict)

            if len(evict_indices) > 0:
                past_key_values = self._remove_tokens(past_key_values, evict_indices)
                for idx in evict_indices.tolist():
                    self._compressed.pop(idx, None)
                self.scorer.reset()

        # 4. Compress moderate-tier tokens (outside recent window, not yet compressed)
        seq_len = past_key_values.layers[0].keys.shape[2]
        scores = self.scorer.score(seq_len)
        compress_start = self.sink_tokens
        compress_end = max(compress_start, seq_len - self.recent_window)
        n_compressible = compress_end - compress_start

        if n_compressible > 0:
            compressible_scores = scores[compress_start:compress_end]
            threshold = compressible_scores.float().quantile(self.compress_ratio)

            for i in range(compress_start, compress_end):
                if i not in self._compressed and compressible_scores[i - compress_start] <= threshold:
                    for layer in past_key_values.layers:
                        token_k = layer.keys[:, :, i:i+1, :]
                        token_v = layer.values[:, :, i:i+1, :]
                        q_k, meta_k = self.compressor.compress(token_k)
                        q_v, meta_v = self.compressor.compress(token_v)
                        layer.keys[:, :, i:i+1, :] = self.compressor.decompress(q_k, meta_k)
                        layer.values[:, :, i:i+1, :] = self.compressor.decompress(q_v, meta_v)
                    self._compressed[i] = True

        return past_key_values

    def _remove_tokens(self, past_key_values, indices: Tensor):
        """Remove tokens at given indices from all layers of the cache."""
        seq_len = past_key_values.layers[0].keys.shape[2]

        for layer in past_key_values.layers:
            device = layer.keys.device
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            keep_mask[indices.to(device)] = False
            layer.keys = layer.keys[:, :, keep_mask, :]
            layer.values = layer.values[:, :, keep_mask, :]

        return past_key_values

    def reset(self) -> None:
        """Reset all state between sequences."""
        self.scorer.reset()
        self._compressed.clear()

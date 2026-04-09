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
    """

    def __init__(
        self,
        memory_budget_gb: float = 4.0,
        recent_window: int = 256,
        compress_dtype: DType = "int8",
        importance_layers: int = 4,
    ):
        self.scorer = ImportanceScorer(num_layers=importance_layers)
        self.compressor = CacheCompressor(dtype=compress_dtype)
        self.eviction_policy = EvictionPolicy(
            memory_budget_gb=memory_budget_gb,
            recent_window=recent_window,
        )
        self.recent_window = recent_window

    def step(
        self,
        past_key_values: tuple,
        attention_weights: list[Tensor],
    ) -> tuple:
        """
        Called after each decoding step to update and potentially
        compress/evict tokens from the KV-cache.

        Args:
            past_key_values:   HuggingFace-style KV-cache tuple.
            attention_weights: Per-layer attention weight tensors,
                               each of shape (batch, heads, 1, seq_len).

        Returns:
            Updated past_key_values after tier management.
        """
        # TODO: implement
        # 1. Update importance scores with new attention weights
        # 2. Estimate current cache memory usage
        # 3. If over budget: evict lowest-scored tokens (outside recent window)
        # 4. Compress moderate-tier tokens
        raise NotImplementedError

    def reset(self) -> None:
        """Reset scorer state between sequences."""
        self.scorer.reset()

"""
Budget-triggered KV-cache eviction policy.
Evicts the lowest-importance tokens when cache memory exceeds a set budget.
"""

import torch
from torch import Tensor


class EvictionPolicy:
    """
    Decides which cached tokens to evict when the memory budget is exceeded.
    Operates on importance scores produced by ImportanceScorer.
    """

    def __init__(self, memory_budget_gb: float, recent_window: int = 256):
        """
        Args:
            memory_budget_gb: Maximum allowed KV-cache memory in GB.
            recent_window:    Number of most-recent tokens that are always
                              protected from eviction.
        """
        self.memory_budget_bytes = int(memory_budget_gb * 1024 ** 3)
        self.recent_window = recent_window

    def should_evict(self, current_cache_bytes: int) -> bool:
        """Return True if the cache exceeds the memory budget."""
        return current_cache_bytes > self.memory_budget_bytes

    def select_eviction_indices(
        self,
        scores: Tensor,
        n_evict: int,
    ) -> Tensor:
        """
        Select indices of tokens to evict.

        Args:
            scores:   Importance scores of shape (seq_len,). Recent tokens
                      (last self.recent_window positions) must be excluded.
            n_evict:  Number of tokens to evict.

        Returns:
            1-D LongTensor of token indices to remove, sorted ascending.
        """
        # TODO: implement — protect recent_window, evict lowest-scored tokens
        raise NotImplementedError

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

    Only tokens outside the recent_window are candidates for eviction.
    Tokens are evicted in ascending score order (least important first).
    """

    def __init__(self, memory_budget_gb: float = 4.0, recent_window: int = 256):
        """
        Args:
            memory_budget_gb: Maximum allowed KV-cache memory in GB.
            recent_window:    Number of most-recent tokens always protected from eviction.
        """
        self.memory_budget_bytes = int(memory_budget_gb * 1024 ** 3)
        self.recent_window = recent_window

    def should_evict(self, current_cache_bytes: int) -> bool:
        """Return True if the cache exceeds the memory budget."""
        return current_cache_bytes > self.memory_budget_bytes

    def n_tokens_to_evict(self, current_cache_bytes: int, bytes_per_token: int) -> int:
        """
        Compute how many tokens need to be evicted to get back under budget.

        Args:
            current_cache_bytes: Current total KV cache size in bytes.
            bytes_per_token:     Memory cost of one token across all layers.

        Returns:
            Number of tokens to evict (0 if already under budget).
        """
        excess = current_cache_bytes - self.memory_budget_bytes
        if excess <= 0:
            return 0
        return max(1, (excess + bytes_per_token - 1) // bytes_per_token)

    def select_eviction_indices(self, scores: Tensor, n_evict: int) -> Tensor:
        """
        Select indices of tokens to evict.

        Protects the last `recent_window` tokens unconditionally.
        Among evictable tokens, selects the n_evict lowest-scored ones.

        Args:
            scores:  Importance scores of shape (seq_len,).
            n_evict: Number of tokens to evict.

        Returns:
            1-D LongTensor of token indices to remove, sorted ascending.
            Empty tensor if nothing can or needs to be evicted.
        """
        seq_len = scores.shape[0]
        n_evictable = max(0, seq_len - self.recent_window)

        if n_evictable == 0 or n_evict <= 0:
            return torch.tensor([], dtype=torch.long)

        n_evict = min(n_evict, n_evictable)

        # Only consider tokens outside the recent window
        evictable_scores = scores[:n_evictable]
        _, sorted_indices = evictable_scores.sort(descending=False)
        evict_indices = sorted_indices[:n_evict]

        return evict_indices.sort().values

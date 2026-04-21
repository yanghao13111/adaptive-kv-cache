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

    Protected zones (never evicted):
    - First `sink_tokens` tokens: attention sinks that absorb excess attention
      mass regardless of content (StreamingLLM). Evicting them causes perplexity
      spikes on long sequences.
    - Last `recent_window` tokens: recency bias — newly generated tokens have not
      yet accumulated enough attention history to score fairly.

    Among the remaining tokens, eviction proceeds in ascending score order.
    """

    def __init__(
        self,
        memory_budget_gb: float = 4.0,
        recent_window: int = 256,
        sink_tokens: int = 4,
    ):
        """
        Args:
            memory_budget_gb: Maximum allowed KV-cache memory in GB.
            recent_window:    Number of most-recent tokens always protected from eviction.
            sink_tokens:      Number of initial prompt tokens to protect as attention sinks.
        """
        self.memory_budget_bytes = int(memory_budget_gb * 1024 ** 3)
        self.recent_window = recent_window
        self.sink_tokens = sink_tokens

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
        # Evictable range: (sink_tokens, seq_len - recent_window)
        n_evictable = max(0, seq_len - self.recent_window - self.sink_tokens)

        if n_evictable == 0 or n_evict <= 0:
            return torch.tensor([], dtype=torch.long)

        n_evict = min(n_evict, n_evictable)

        evictable_scores = scores[self.sink_tokens : seq_len - self.recent_window]
        _, sorted_indices = evictable_scores.sort(descending=False)
        # Offset indices back to full sequence position
        evict_indices = sorted_indices[:n_evict] + self.sink_tokens

        return evict_indices.sort().values

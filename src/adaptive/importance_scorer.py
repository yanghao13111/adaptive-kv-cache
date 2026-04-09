"""
Attention-based importance scoring for KV-cache tokens.
Scores are derived from accumulated attention weights across recent layers.
"""

import torch
from torch import Tensor


class ImportanceScorer:
    """
    Scores each cached token by how much attention it has received
    over the last L decoder layers.

    Higher score → token is more important → keep in cache.
    """

    def __init__(self, num_layers: int = 4):
        """
        Args:
            num_layers: Number of recent layers whose attention weights
                        are aggregated for the importance score.
        """
        self.num_layers = num_layers
        # attention_accumulator[layer_idx] -> Tensor of shape (seq_len,)
        self._attention_accumulator: list[Tensor] = []

    def update(self, attention_weights: Tensor, layer_idx: int) -> None:
        """
        Ingest attention weights from one decoding step / one layer.

        Args:
            attention_weights: shape (batch, heads, 1, seq_len) — weights
                               for the newly generated token over all cached tokens.
            layer_idx: which layer these weights come from.
        """
        # TODO: implement
        raise NotImplementedError

    def score(self, seq_len: int) -> Tensor:
        """
        Return an importance score for each of the seq_len cached tokens.

        Returns:
            Tensor of shape (seq_len,), higher = more important.
        """
        # TODO: implement
        raise NotImplementedError

    def reset(self) -> None:
        """Clear accumulated attention statistics (call between sequences)."""
        self._attention_accumulator = []

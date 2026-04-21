"""
Attention-based importance scoring for KV-cache tokens.
Scores are derived from accumulated attention weights across all layers.
"""

import torch
from torch import Tensor


class ImportanceScorer:
    """
    Scores each cached token by how much attention it has received
    across all decoder layers and all decoding steps so far.

    Higher score → token is more important → keep in cache.

    Score is the cumulative sum of attention weights this token received
    as a key, averaged across all attention heads, summed across all layers.
    """

    def __init__(self, num_layers: int = 12):
        """
        Args:
            num_layers: Total number of transformer layers in the model.
        """
        self.num_layers = num_layers
        # Accumulated importance scores, shape (seq_len,). Grows as new tokens are added.
        self._scores: Tensor | None = None

    def update(self, attentions: tuple[Tensor, ...]) -> None:
        """
        Update importance scores from one full decoding step.

        Args:
            attentions: tuple of length num_layers, each tensor has shape
                        (batch, heads, query_len, key_len).
                        query_len is 1 for autoregressive decoding (one new token).
                        key_len is the current cache length.
        """
        # Sum attention across all layers and average across heads.
        # Each layer contributes equally to the importance score.
        # Result shape: (key_len,)
        step_scores = None
        for layer_attn in attentions:
            if layer_attn is None:
                continue
            # layer_attn: (batch, heads, query_len, key_len) — take last query position
            layer_scores = layer_attn[0, :, -1, :].mean(dim=0)  # (key_len,)
            if step_scores is None:
                step_scores = layer_scores
            else:
                step_scores = step_scores + layer_scores

        if step_scores is None:
            return

        if self._scores is None:
            self._scores = step_scores
        else:
            # Cache grew by one token (the newly generated one starts with score 0)
            current_len = step_scores.shape[0]
            if self._scores.shape[0] < current_len:
                padding = torch.zeros(
                    current_len - self._scores.shape[0],
                    device=self._scores.device,
                    dtype=self._scores.dtype,
                )
                self._scores = torch.cat([self._scores, padding], dim=0)

            self._scores = self._scores + step_scores

    def score(self, seq_len: int) -> Tensor:
        """
        Return importance scores for the current cached tokens.

        Args:
            seq_len: Current cache length.

        Returns:
            Tensor of shape (seq_len,), higher = more important.
            Returns uniform scores if no updates have been recorded yet.
        """
        if self._scores is None:
            return torch.ones(seq_len)

        # Trim or pad to match current seq_len
        if self._scores.shape[0] >= seq_len:
            return self._scores[:seq_len]
        else:
            padding = torch.zeros(
                seq_len - self._scores.shape[0],
                device=self._scores.device,
                dtype=self._scores.dtype,
            )
            return torch.cat([self._scores, padding], dim=0)

    def reset(self) -> None:
        """Clear accumulated scores (call between sequences)."""
        self._scores = None

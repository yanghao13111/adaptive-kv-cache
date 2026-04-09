"""
Low-precision KV-cache compression.
Quantizes key/value tensors to INT8 or INT4 to reduce memory usage.
"""

import torch
from torch import Tensor
from typing import Literal


DType = Literal["int8", "int4"]


class CacheCompressor:
    """
    Compresses KV tensors to a lower precision dtype and decompresses
    them back to FP16/BF16 before attention computation.
    """

    def __init__(self, dtype: DType = "int8"):
        """
        Args:
            dtype: Target quantization dtype ("int8" or "int4").
        """
        self.dtype = dtype

    def compress(self, tensor: Tensor) -> tuple[Tensor, dict]:
        """
        Quantize a key or value tensor.

        Args:
            tensor: FP16/BF16 tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            (quantized_tensor, metadata) where metadata holds scale/zero-point
            needed for decompression.
        """
        # TODO: implement per-channel or per-token quantization
        raise NotImplementedError

    def decompress(self, quantized: Tensor, metadata: dict) -> Tensor:
        """
        Dequantize back to the original floating-point dtype.

        Args:
            quantized: Quantized tensor from compress().
            metadata:  Scale/zero-point dict from compress().

        Returns:
            FP16/BF16 tensor of the original shape.
        """
        # TODO: implement
        raise NotImplementedError

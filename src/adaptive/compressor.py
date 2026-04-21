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
    Compresses KV tensors to lower precision and decompresses them back
    to FP16/BF16 before attention computation.

    Uses per-token quantization: each token's key/value vector gets its
    own scale and zero-point, which preserves accuracy better than
    per-tensor quantization across the whole cache.
    """

    def __init__(self, dtype: DType = "int8"):
        """
        Args:
            dtype: Target quantization dtype ("int8" or "int4").
        """
        self.dtype = dtype
        if dtype == "int8":
            self.qmin, self.qmax = -128, 127
            self.torch_dtype = torch.int8
        elif dtype == "int4":
            # INT4 is simulated via int8 with [-8, 7] range
            self.qmin, self.qmax = -8, 7
            self.torch_dtype = torch.int8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def compress(self, tensor: Tensor) -> tuple[Tensor, dict]:
        """
        Quantize a key or value tensor using per-token min-max scaling.

        Args:
            tensor: FP16/BF16 tensor of shape (batch, heads, seq_len, head_dim).

        Returns:
            (quantized_tensor, metadata) where metadata holds scale and zero_point
            needed for decompression. quantized_tensor has the same shape but
            stored as int8.
        """
        original_dtype = tensor.dtype
        tensor_fp32 = tensor.float()  # quantize in FP32 for numerical stability

        # Per-token quantization: compute scale per (batch, heads, seq_len) vector
        # Shape of min/max: (batch, heads, seq_len, 1)
        t_min = tensor_fp32.amin(dim=-1, keepdim=True)
        t_max = tensor_fp32.amax(dim=-1, keepdim=True)

        # Avoid division by zero for constant vectors
        scale = (t_max - t_min).clamp(min=1e-8) / (self.qmax - self.qmin)
        zero_point = self.qmin - (t_min / scale)
        zero_point = zero_point.round().clamp(self.qmin, self.qmax)

        quantized = ((tensor_fp32 / scale) + zero_point).round().clamp(self.qmin, self.qmax)
        quantized = quantized.to(self.torch_dtype)

        metadata = {
            "scale": scale,
            "zero_point": zero_point,
            "original_dtype": original_dtype,
        }

        return quantized, metadata

    def decompress(self, quantized: Tensor, metadata: dict) -> Tensor:
        """
        Dequantize back to the original floating-point dtype.

        Args:
            quantized: INT8 tensor from compress().
            metadata:  Dict with scale, zero_point, original_dtype from compress().

        Returns:
            Tensor restored to original dtype and approximate values.
        """
        scale = metadata["scale"]
        zero_point = metadata["zero_point"]
        original_dtype = metadata["original_dtype"]

        dequantized = (quantized.float() - zero_point) * scale
        return dequantized.to(original_dtype)

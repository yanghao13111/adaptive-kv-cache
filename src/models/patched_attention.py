"""
HuggingFace attention forward-pass hook.
Patches model attention layers to intercept KV-cache and attention weights,
enabling the AdaptiveCacheManager to observe and modify them at each step.
"""

import torch
from transformers import AutoModelForCausalLM
from src.adaptive.cache_manager import AdaptiveCacheManager


def patch_model(
    model: AutoModelForCausalLM,
    cache_manager: AdaptiveCacheManager,
) -> AutoModelForCausalLM:
    """
    Register forward hooks on all attention layers of `model` to:
      1. Capture per-layer attention weights after each decoding step.
      2. Pass them to `cache_manager.step()` for tier management.

    Args:
        model:         A HuggingFace causal LM (e.g. LlamaForCausalLM).
        cache_manager: The AdaptiveCacheManager instance to use.

    Returns:
        The same model with hooks attached (in-place modification).
    """
    # TODO: implement
    # Hint: use model.register_forward_hook or per-layer hooks on
    # model.model.layers[i].self_attn
    raise NotImplementedError


def unpatch_model(model: AutoModelForCausalLM) -> None:
    """Remove all registered hooks from the model."""
    # TODO: implement
    raise NotImplementedError

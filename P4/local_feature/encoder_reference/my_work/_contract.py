"""Shared input contract; no preprocessing or architecture is hidden here."""

import torch


def validate_input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError("images must have shape [B, 3, H, W]")
    if x.shape[1] != 3:
        raise ValueError("Expected 3 channels in NIR, red, green order")
    if not x.is_floating_point():
        raise TypeError("images must be floating point; use normalized float32 inputs")
    if x.shape[0] < 1 or any(d < 32 or d % 32 for d in x.shape[-2:]):
        raise ValueError("Batch must be nonempty; H and W must be positive multiples of 32")


def validate_weights(weights):
    if weights not in (None, "imagenet"):
        raise ValueError("weights must be None or 'imagenet'")

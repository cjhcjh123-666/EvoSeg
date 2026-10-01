"""Small, explicitly gated CPG-VIRST SFT losses."""

from __future__ import annotations

import torch
from torch import Tensor


def object_discrimination_loss(scores: Tensor, target_index: Tensor) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("object scores must have shape [batch, objects]")
    if scores.shape[1] < 2:
        return scores.sum() * 0.0
    return torch.nn.functional.cross_entropy(scores, target_index)


def soft_dice_loss(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    intersection = (prediction * target).sum(dim=-1)
    denominator = prediction.sum(dim=-1) + target.sum(dim=-1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def interval_localization_loss(posterior: Tensor, target: Tensor) -> Tensor:
    if posterior.shape != target.shape:
        raise ValueError("interval posterior and target must have identical shape")
    bce = torch.nn.functional.binary_cross_entropy(
        posterior.clamp(1e-6, 1.0 - 1e-6), target.float()
    )
    return bce + soft_dice_loss(posterior, target)


def order_contrast_loss(original: Tensor, permuted: Tensor, margin: float = 0.2) -> Tensor:
    return torch.relu(margin - original + permuted).mean()


def termination_entropy_regularizer(probability: Tensor) -> Tensor:
    """Negative entropy: minimizing it weakly discourages N=1 collapse."""

    value = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    return (value * value.log()).sum(dim=-1).mean()

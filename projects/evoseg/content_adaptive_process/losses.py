"""Preregistered CAPG-v2 SFT objectives."""

from __future__ import annotations

import torch
from torch import Tensor


def soft_dice_loss(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    intersection = (prediction * target).sum(dim=-1)
    denominator = prediction.sum(dim=-1) + target.sum(dim=-1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def process_envelope_loss(occupancy: Tensor, target: Tensor) -> Tensor:
    if occupancy.shape != target.shape:
        raise ValueError("occupancy and process envelope must have identical shape")
    probability = occupancy.float().clamp(1e-6, 1.0 - 1e-6)
    truth = target.float()
    # Explicit probability-space BCE is autocast-safe; the standard BCE
    # wrapper intentionally rejects autocast even after an explicit FP32 cast.
    bce = -(truth * probability.log() + (1.0 - truth) * (1.0 - probability).log()).mean()
    return bce + soft_dice_loss(probability, truth)


def temporal_iou(occupancy: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    prediction = occupancy >= threshold
    truth = target.to(torch.bool)
    intersection = (prediction & truth).sum(dim=-1).float()
    union = (prediction | truth).sum(dim=-1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def object_discrimination_loss(scores: Tensor, target_index: Tensor) -> Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        return scores.sum() * 0.0
    return torch.nn.functional.cross_entropy(scores, target_index)


def order_contrast_loss(original: Tensor, permuted: Tensor, margin: float = 0.2) -> Tensor:
    return torch.relu(margin - original + permuted).mean()


def language_order_loss(
    positions: Tensor, length_prior: Tensor, margin: float = 0.03
) -> Tensor:
    if positions.shape != length_prior.shape:
        raise ValueError("positions and length prior must have identical shape")
    # An adjacent pair m,m+1 is active iff N >= m+2.
    active = torch.flip(
        torch.cumsum(torch.flip(length_prior, dims=[-1]), dim=-1), dims=[-1]
    )[:, 1:]
    violation = torch.relu(margin - (positions[:, 1:] - positions[:, :-1]))
    return (violation * active).sum() / active.sum().clamp_min(1e-6)


def language_overlap_loss(attention: Tensor, length_prior: Tensor) -> Tensor:
    left = torch.nn.functional.normalize(attention[:, :-1], dim=-1)
    right = torch.nn.functional.normalize(attention[:, 1:], dim=-1)
    active = torch.flip(
        torch.cumsum(torch.flip(length_prior, dims=[-1]), dim=-1), dims=[-1]
    )[:, 1:]
    overlap = (left * right).sum(dim=-1)
    return (overlap * active).sum() / active.sum().clamp_min(1e-6)


def segment_alignment_loss(
    process_states: Tensor,
    frame_features: Tensor,
    posterior_process: Tensor,
    length_prior: Tensor,
    temperature: float = 0.07,
    detach_posterior: bool = False,
) -> tuple[Tensor, Tensor]:
    gamma = posterior_process.detach() if detach_posterior else posterior_process
    denominator = gamma.sum(dim=1).clamp_min(1e-6)
    pooled = torch.einsum("btm,btd->bmd", gamma, frame_features) / denominator.unsqueeze(-1)
    language = torch.nn.functional.normalize(process_states, dim=-1)
    visual = torch.nn.functional.normalize(pooled, dim=-1)
    logits = torch.einsum("bmd,bnd->bmn", language, visual) / temperature
    batch, states, _ = logits.shape
    labels = torch.arange(states, device=logits.device).view(1, states).expand(batch, -1)
    per_state = torch.nn.functional.cross_entropy(
        logits.reshape(batch * states, states), labels.reshape(-1), reduction="none"
    ).reshape(batch, states)
    active = torch.flip(
        torch.cumsum(torch.flip(length_prior, dims=[-1]), dim=-1), dims=[-1]
    )
    loss = (per_state * active).sum() / active.sum().clamp_min(1e-6)
    diagonal = logits.diagonal(dim1=-2, dim2=-1)
    masked = logits.masked_fill(torch.eye(states, device=logits.device, dtype=torch.bool), -torch.inf)
    margin = diagonal - masked.max(dim=-1).values
    semantic_margin = (margin * active).sum() / active.sum().clamp_min(1e-6)
    return loss, semantic_margin

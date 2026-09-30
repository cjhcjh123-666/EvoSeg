"""Discriminative Ordered Process Grounding v2 on frozen features."""
from __future__ import annotations

import math

import torch
from torch import nn

from projects.evoseg.ordered_process_grounding.model import (
    ProcessCompiler,
    monotonic_logsumexp,
)


def candidate_standardize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize over candidates without making a one-candidate query invalid."""
    if values.shape[-1] < 2:
        return torch.zeros_like(values)
    return (values - values.mean(dim=-1, keepdim=True)) / (
        values.std(dim=-1, keepdim=True, unbiased=False) + eps
    )


class DiscriminativeOrderedHead(nn.Module):
    """Ordered scorer with candidate-relative, non-silent calibrated fusion."""

    def __init__(
        self,
        query_dim: int = 2560,
        visual_dim: int = 1152,
        hidden_dim: int = 256,
        slots: int = 4,
        alignment: str = "monotonic",
        use_delta_fusion: bool = True,
    ):
        super().__init__()
        if alignment not in {"monotonic", "global"}:
            raise ValueError(f"unsupported alignment: {alignment}")
        self.alignment = alignment
        self.use_delta_fusion = use_delta_fusion
        self.compiler = ProcessCompiler(query_dim, hidden_dim, slots)
        self.track_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU()
        )
        self.slot_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        initial = math.log(math.expm1(1.0))
        self.beta_order_raw = nn.Parameter(torch.tensor(initial))
        self.beta_delta_raw = nn.Parameter(torch.tensor(initial))

    @property
    def beta_order(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.beta_order_raw)

    @property
    def beta_delta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.beta_delta_raw)

    def order_scores(
        self,
        query_tokens: torch.Tensor,
        tracks: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slots, gates = self.compiler(query_tokens, token_mask)
        slots = torch.nn.functional.normalize(self.slot_projection(slots), dim=-1)
        track_values = torch.nn.functional.normalize(self.track_projection(tracks), dim=-1)
        similarities = torch.einsum("bmh,bcth->bcmt", slots, track_values)
        gated = similarities * gates[:, None, :, None]
        denominator = gates.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        if self.alignment == "monotonic":
            score = monotonic_logsumexp(gated) / denominator
        else:
            score = torch.logsumexp(gated, dim=-1).sum(dim=-1) / denominator
        return score, {"slots": slots, "gates": gates, "similarities": similarities}

    def forward(
        self,
        base_scores: torch.Tensor,
        query_tokens: torch.Tensor,
        original_tracks: torch.Tensor,
        permuted_tracks: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original, audit = self.order_scores(query_tokens, original_tracks, token_mask)
        permuted, _ = self.order_scores(query_tokens, permuted_tracks, token_mask)
        delta = original - permuted
        base_z = candidate_standardize(base_scores)
        order_z = candidate_standardize(original)
        delta_z = candidate_standardize(delta)
        final = base_z + self.beta_order * order_z
        if self.use_delta_fusion:
            final = final + self.beta_delta * delta_z
        return final, {
            **audit,
            "base": base_scores,
            "base_z": base_z,
            "order": original,
            "order_permuted": permuted,
            "order_z": order_z,
            "delta": delta,
            "delta_z": delta_z,
            "beta_order": self.beta_order,
            "beta_delta": self.beta_delta,
        }

"""Small residual ordered-process scorer on frozen query/track features."""
from __future__ import annotations

import torch
from torch import nn


_MONOTONIC_SELECTOR_CACHE: dict[
    tuple[int, int, str, int | None, torch.dtype], torch.Tensor
] = {}


def _monotonic_selector(
    slots: int, steps: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a cached matrix that sums every strictly ordered alignment."""
    key = (slots, steps, device.type, device.index, dtype)
    selector = _MONOTONIC_SELECTOR_CACHE.get(key)
    if selector is None:
        paths = torch.combinations(torch.arange(steps, device=device), r=slots)
        selector = torch.zeros(
            len(paths), slots * steps, device=device, dtype=dtype
        )
        columns = torch.arange(slots, device=device)[None, :] * steps + paths
        selector[torch.arange(len(paths), device=device)[:, None], columns] = 1
        _MONOTONIC_SELECTOR_CACHE[key] = selector
    return selector


def monotonic_logsumexp(scores: torch.Tensor) -> torch.Tensor:
    """Soft score of all strictly monotonic alignments.

    Args:
        scores: ``[..., process_slot, time]``.  Time must be at least the
            number of slots.  Every valid path selects one strictly increasing
            time position per slot.
    """
    if scores.ndim < 2:
        raise ValueError("scores must end in [slot, time]")
    slots, steps = scores.shape[-2:]
    if steps < slots:
        raise ValueError(f"strict alignment needs time >= slots ({steps} < {slots})")
    # M=4,T=8 has only 70 legal paths.  Explicitly enumerating those paths is
    # mathematically the same log-sum-exp DP, while avoiding undefined
    # gradients from all-``-inf`` impossible prefix states.
    selector = _monotonic_selector(slots, steps, scores.device, scores.dtype)
    path_values = scores.flatten(-2) @ selector.transpose(0, 1)
    return torch.logsumexp(path_values, dim=-1)


class ProcessCompiler(nn.Module):
    """Compile an unsegmented query-token sequence into ordered slots."""

    def __init__(self, query_dim: int = 2560, hidden_dim: int = 256, slots: int = 4):
        super().__init__()
        self.slots = slots
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden_dim), nn.GELU()
        )
        self.process_slots = nn.Parameter(torch.empty(slots, hidden_dim))
        self.ordinal = nn.Parameter(torch.empty(slots, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=0.0
        )
        self.validity = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.process_slots, std=0.02)
        nn.init.normal_(self.ordinal, std=0.02)

    def forward(self, query_tokens: torch.Tensor, token_mask: torch.Tensor | None = None):
        # query_tokens: [batch, token, query_dim]
        tokens = self.query_projection(query_tokens)
        slots = (self.process_slots + self.ordinal).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        values, _ = self.cross_attention(
            slots, tokens, tokens,
            key_padding_mask=(~token_mask.bool()) if token_mask is not None else None,
            need_weights=False,
        )
        gates = torch.sigmoid(self.validity(values)).squeeze(-1)
        return values, gates


class OrderedResidualHead(nn.Module):
    """Candidate order score; spatial output always remains the candidate mask."""

    def __init__(
        self,
        query_dim: int = 2560,
        visual_dim: int = 1152,
        hidden_dim: int = 256,
        slots: int = 4,
        alignment: str = "monotonic",
    ):
        super().__init__()
        if alignment not in {"monotonic", "global"}:
            raise ValueError(f"unsupported alignment: {alignment}")
        self.alignment = alignment
        self.compiler = ProcessCompiler(query_dim, hidden_dim, slots)
        self.track_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU()
        )
        self.slot_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(-4.59511985013459))  # sigmoid == 0.01

    def order_scores(
        self,
        query_tokens: torch.Tensor,
        tracks: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return order scores for tracks ``[batch,candidate,time,visual]``."""
        slots, gates = self.compiler(query_tokens, token_mask)
        slots = torch.nn.functional.normalize(self.slot_projection(slots), dim=-1)
        track_values = torch.nn.functional.normalize(self.track_projection(tracks), dim=-1)
        # Cosine similarity is already dimension-normalized.  An additional
        # sqrt(hidden) divisor would make the near-zero residual numerically
        # unable to grow during the preregistered short SFT budget.
        similarities = torch.einsum("bmh,bcth->bcmt", slots, track_values)
        gated = similarities * gates[:, None, :, None]
        denominator = gates.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        if self.alignment == "monotonic":
            score = monotonic_logsumexp(gated) / denominator
        else:
            # Same compiler/projections, but no temporal-order constraint.
            score = torch.logsumexp(gated, dim=-1).sum(dim=-1) / denominator
        return score, {"slots": slots, "gates": gates, "similarities": similarities}

    def forward(
        self,
        base_scores: torch.Tensor,
        query_tokens: torch.Tensor,
        tracks: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        order, audit = self.order_scores(query_tokens, tracks, token_mask)
        alpha = torch.sigmoid(self.alpha)
        return base_scores + alpha * order, order, {**audit, "alpha": alpha}


class TokenMeanPoolScorer(nn.Module):
    """Order-free GroundMoRe base when no prior dataset-specific base exists."""

    def __init__(self, query_dim: int = 2560, visual_dim: int = 1152, hidden_dim: int = 256):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden_dim), nn.GELU())
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim), nn.GELU())
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, query_tokens: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        # query_tokens [token,Q], tracks [candidate,time,V]
        query = self.query(query_tokens).mean(dim=0, keepdim=True)
        visual = self.visual(tracks).mean(dim=1)
        return self.score(query + visual).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

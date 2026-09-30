"""Small ordered-process module inserted before VIRST's SegPrompter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .monotonic import MonotonicAlignment


@dataclass
class ProcessConditionerOutput:
    frame_states: Tensor
    process_slots: Tensor
    validity: Tensor
    compatibility: Tensor
    alignment: Tensor
    alignment_score: Tensor
    beta: Tensor


class ProcessCompiler(nn.Module):
    def __init__(self, query_dim: int, process_dim: int = 256, slots: int = 4, heads: int = 8) -> None:
        super().__init__()
        self.slots = slots
        self.query_proj = nn.Linear(query_dim, process_dim)
        self.process_queries = nn.Parameter(torch.empty(slots, process_dim))
        self.ordinal = nn.Parameter(torch.empty(slots, process_dim))
        self.attn = nn.MultiheadAttention(process_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(process_dim)
        self.validity = nn.Linear(process_dim, 1)
        nn.init.normal_(self.process_queries, std=0.02)
        nn.init.normal_(self.ordinal, std=0.02)

    def forward(self, query_tokens: Tensor, query_padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch = query_tokens.shape[0]
        memory = self.query_proj(query_tokens)
        slots = (self.process_queries + self.ordinal).unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.attn(
            slots,
            memory,
            memory,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        slots = self.norm(slots + attended)
        validity = self.validity(slots).squeeze(-1).sigmoid()
        return slots, validity


class ProcessConditioner(nn.Module):
    """Compile language process slots and align them to VIRST video features.

    ``frame_states`` are residuals in SegPrompter token space. They are meant to
    be added to the per-frame repeated [SEG] state *before* its decoder layers.
    """

    def __init__(
        self,
        query_dim: int,
        vision_dim: int = 256,
        process_dim: int = 256,
        slots: int = 4,
        heads: int = 8,
        alignment_mode: str = "monotonic",
    ) -> None:
        super().__init__()
        if alignment_mode not in {"monotonic", "global"}:
            raise ValueError(f"unsupported alignment mode: {alignment_mode}")
        self.alignment_mode = alignment_mode
        self.compiler = ProcessCompiler(query_dim, process_dim, slots, heads)
        self.spatial_down = nn.Sequential(
            nn.Conv2d(vision_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.visual_norm = nn.LayerNorm(process_dim)
        self.slot_proj = nn.Linear(process_dim, process_dim, bias=False)
        self.visual_proj = nn.Linear(process_dim, process_dim, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(process_dim, process_dim),
            nn.GELU(),
            nn.Linear(process_dim, process_dim),
        )
        self.aligner = MonotonicAlignment()
        # softplus(raw_beta) == 1 at initialization.
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))

    @property
    def beta(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_beta)

    def forward(
        self,
        query_tokens: Tensor,
        video_features: Tensor,
        query_padding_mask: Tensor | None = None,
    ) -> ProcessConditionerOutput:
        if video_features.ndim != 5:
            raise ValueError("video_features must have shape [batch, frames, channels, height, width]")
        batch, frames, channels, height, width = video_features.shape
        slots, validity = self.compiler(query_tokens, query_padding_mask)

        spatial = self.spatial_down(video_features.reshape(batch * frames, channels, height, width))
        spatial = spatial.flatten(2).transpose(1, 2).reshape(batch, frames, -1, slots.shape[-1])
        spatial = self.visual_norm(spatial)
        q = self.slot_proj(slots)
        kv = self.visual_proj(spatial)
        spatial_logits = torch.einsum("bmd,btsd->bmts", q, kv) / math.sqrt(q.shape[-1])
        spatial_attention = spatial_logits.softmax(dim=-1)
        event_frame = torch.einsum("bmts,btsd->bmtd", spatial_attention, spatial)
        compatibility = torch.einsum("bmd,bmtd->bmt", q, event_frame) / math.sqrt(q.shape[-1])
        if self.alignment_mode == "monotonic":
            aligned = self.aligner(compatibility, validity)
            alignment = aligned.posterior
            alignment_score = aligned.score
        else:
            alignment = compatibility.softmax(dim=-1) * validity.unsqueeze(-1)
            alignment_score = (
                validity * torch.logsumexp(compatibility, dim=-1)
            ).sum(dim=-1)
        frame_context = torch.einsum("bmt,bmtd->btd", alignment, event_frame)
        frame_states = self.beta * self.adapter(frame_context)
        return ProcessConditionerOutput(
            frame_states=frame_states,
            process_slots=slots,
            validity=validity,
            compatibility=compatibility,
            alignment=alignment,
            alignment_score=alignment_score,
            beta=self.beta,
        )

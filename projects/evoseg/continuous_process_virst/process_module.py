"""Continuous process compiler and VIRST spatial/process alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .segmental import SegmentalAlignmentOutput, SegmentalForwardBackward


@dataclass
class ContinuousProcessOutput:
    frame_states: Tensor
    process_states: Tensor
    compatibility: Tensor
    alignment: SegmentalAlignmentOutput
    process_confidence: Tensor
    beta: Tensor


class ProcessChainCompiler(nn.Module):
    def __init__(
        self, query_dim: int, process_dim: int = 256, max_states: int = 6, heads: int = 8
    ) -> None:
        super().__init__()
        self.max_states = max_states
        self.input_norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, process_dim)
        self.process_queries = nn.Parameter(torch.empty(max_states, process_dim))
        self.ordinal = nn.Parameter(torch.empty(max_states, process_dim))
        self.cross_attention = nn.MultiheadAttention(process_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(process_dim)
        self.termination = nn.Linear(process_dim, 1)
        nn.init.normal_(self.process_queries, std=0.02)
        nn.init.normal_(self.ordinal, std=0.02)

    def forward(
        self, query_tokens: Tensor, query_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch = query_tokens.shape[0]
        memory = self.query_projection(self.input_norm(query_tokens))
        queries = (self.process_queries + self.ordinal).unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        states = self.output_norm(queries + attended)
        termination_logits = self.termination(states).squeeze(-1)
        return states, termination_logits


class ContinuousProcessConditioner(nn.Module):
    def __init__(
        self,
        query_dim: int,
        vision_dim: int = 256,
        process_dim: int = 256,
        max_states: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.compiler = ProcessChainCompiler(
            query_dim, process_dim, max_states=max_states, heads=heads
        )
        self.spatial_down = nn.Sequential(
            nn.Conv2d(vision_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.spatial_norm = nn.LayerNorm(process_dim)
        self.process_projection = nn.Linear(process_dim, process_dim, bias=False)
        self.visual_projection = nn.Linear(process_dim, process_dim, bias=False)
        self.alignment = SegmentalForwardBackward(max_states)
        self.context_projection = nn.Linear(process_dim, process_dim)
        self.adapter = nn.Sequential(
            nn.Linear(process_dim + 1, process_dim),
            nn.GELU(),
            nn.Linear(process_dim, process_dim),
        )
        # softplus(raw_beta) == 0.5 at initialization.
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

    @property
    def beta(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_beta)

    def forward(
        self,
        query_tokens: Tensor,
        video_features: Tensor,
        query_padding_mask: Tensor | None = None,
    ) -> ContinuousProcessOutput:
        if video_features.ndim != 5:
            raise ValueError("video_features must have shape [batch, frames, channels, H, W]")
        batch, frames, channels, height, width = video_features.shape
        process, termination_logits = self.compiler(query_tokens, query_padding_mask)
        spatial = self.spatial_down(
            video_features.reshape(batch * frames, channels, height, width)
        )
        spatial = spatial.flatten(2).transpose(1, 2)
        spatial = spatial.reshape(batch, frames, spatial.shape[1], spatial.shape[2])
        spatial = self.spatial_norm(spatial)
        query = self.process_projection(process)
        visual = self.visual_projection(spatial)
        logits = torch.einsum("bmd,btsd->bmts", query, visual) / math.sqrt(query.shape[-1])
        attention = logits.softmax(dim=-1)
        event_frame = torch.einsum("bmts,btsd->bmtd", attention, spatial)
        compatibility = torch.einsum("bmd,bmtd->bmt", query, event_frame) / math.sqrt(
            query.shape[-1]
        )
        alignment = self.alignment(compatibility, termination_logits)
        context = torch.einsum(
            "btm,bmd->btd", alignment.posterior, self.context_projection(process)
        )
        confidence = alignment.posterior.max(dim=-1).values
        frame_states = self.beta * self.adapter(
            torch.cat([context, confidence.unsqueeze(-1)], dim=-1)
        )
        return ContinuousProcessOutput(
            frame_states=frame_states,
            process_states=process,
            compatibility=compatibility,
            alignment=alignment,
            process_confidence=confidence,
            beta=self.beta,
        )

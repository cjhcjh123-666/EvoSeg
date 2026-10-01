"""Language-anchored, content-adaptive process conditioning for VIRST."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .segmental import (
    AdaptiveAlignmentOutput,
    AdaptiveSegmentalForwardBackward,
    stick_breaking_probabilities,
)


@dataclass
class ContentAdaptiveProcessOutput:
    frame_states: Tensor
    process_states: Tensor
    language_attention: Tensor
    language_positions: Tensor
    compatibility: Tensor
    alignment: AdaptiveAlignmentOutput
    halt_probabilities: Tensor
    length_prior: Tensor
    beta: Tensor
    spatial_map: Tensor
    frame_features: Tensor
    transition_mode: str


class LanguageAnchoredCompiler(nn.Module):
    def __init__(self, query_dim: int, process_dim: int = 256, max_states: int = 6) -> None:
        super().__init__()
        self.max_states = max_states
        self.input_norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, process_dim)
        self.attention_queries = nn.Parameter(torch.empty(max_states, process_dim))
        self.ordinal = nn.Parameter(torch.empty(max_states, process_dim))
        self.residual = nn.Sequential(
            nn.Linear(process_dim, process_dim), nn.GELU(), nn.Linear(process_dim, process_dim)
        )
        self.output_norm = nn.LayerNorm(process_dim)
        self.halt = nn.Sequential(
            nn.Linear(process_dim * 2, process_dim), nn.GELU(), nn.Linear(process_dim, 1)
        )
        nn.init.normal_(self.attention_queries, std=0.02)
        nn.init.normal_(self.ordinal, std=0.02)

    def forward(
        self, query_tokens: Tensor, query_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, tokens, _ = query_tokens.shape
        memory = self.query_projection(self.input_norm(query_tokens))
        logits = torch.einsum("md,bld->bml", self.attention_queries, memory) / math.sqrt(
            memory.shape[-1]
        )
        if query_padding_mask is not None:
            logits = logits.masked_fill(query_padding_mask.unsqueeze(1), -torch.inf)
        attention = logits.softmax(dim=-1)
        anchor = torch.einsum("bml,bld->bmd", attention, memory)
        states = self.output_norm(anchor + self.ordinal.unsqueeze(0) + self.residual(anchor))
        valid = (
            (~query_padding_mask).to(memory.dtype)
            if query_padding_mask is not None
            else memory.new_ones(batch, tokens)
        )
        q_global = (memory * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        halt_logits = self.halt(
            torch.cat([states, q_global.unsqueeze(1).expand(-1, self.max_states, -1)], dim=-1)
        ).squeeze(-1)
        length_prior, halt_probability = stick_breaking_probabilities(halt_logits)
        positions = torch.linspace(0.0, 1.0, tokens, device=memory.device, dtype=memory.dtype)
        language_positions = (attention * positions.view(1, 1, tokens)).sum(dim=-1)
        return states, attention, language_positions, length_prior, halt_probability


class ContentAdaptiveProcessConditioner(nn.Module):
    def __init__(
        self,
        query_dim: int,
        vision_dim: int = 256,
        process_dim: int = 256,
        max_states: int = 6,
        transition_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if transition_mode not in {"adaptive", "global"}:
            raise ValueError("transition_mode must be adaptive or global")
        self.max_states = max_states
        self.transition_mode = transition_mode
        self.compiler = LanguageAnchoredCompiler(query_dim, process_dim, max_states)
        self.spatial_down = nn.Sequential(
            nn.Conv2d(vision_dim, process_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(process_dim, process_dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.spatial_norm = nn.LayerNorm(process_dim)
        self.process_projection = nn.Linear(process_dim, process_dim, bias=False)
        self.visual_projection = nn.Linear(process_dim, process_dim, bias=False)
        hidden = process_dim
        self.transition_mlp = nn.Sequential(
            nn.Linear(process_dim * 5, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        self.start_mlp = nn.Sequential(
            nn.Linear(process_dim * 3, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(process_dim * 3, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        self.global_start = nn.Parameter(torch.zeros(()))
        self.global_transition = nn.Parameter(torch.zeros(max_states, 2))
        self.global_end = nn.Parameter(torch.zeros(max_states, 2))
        self.alignment = AdaptiveSegmentalForwardBackward(max_states)
        self.context_projection = nn.Linear(process_dim, process_dim)
        self.adapter = nn.Sequential(
            nn.Linear(process_dim + 2, process_dim), nn.GELU(), nn.Linear(process_dim, process_dim)
        )
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

    @property
    def beta(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_beta)

    def _transition_logits(
        self, process: Tensor, frame_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, frames, dim = frame_features.shape
        if self.transition_mode == "global":
            start = self.global_start.expand(batch, frames)
            transition = self.global_transition.view(1, self.max_states, 1, 2).expand(
                batch, -1, max(0, frames - 1), -1
            )
            end = self.global_end.view(1, self.max_states, 1, 2).expand(
                batch, -1, max(0, frames - 1), -1
            )
            return start, transition, end
        delta_at = frame_features.new_zeros(batch, frames, dim)
        if frames > 1:
            delta_at[:, 1:] = frame_features[:, 1:] - frame_features[:, :-1]
        first = process[:, 0].unsqueeze(1).expand(-1, frames, -1)
        start = self.start_mlp(torch.cat([first, frame_features, delta_at], dim=-1)).squeeze(-1)
        if frames == 1:
            empty = frame_features.new_empty(batch, self.max_states, 0, 2)
            return start, empty, empty
        current = process.unsqueeze(2).expand(-1, -1, frames - 1, -1)
        following_process = torch.cat([process[:, 1:], process[:, -1:]], dim=1)
        following_process = following_process.unsqueeze(2).expand(-1, -1, frames - 1, -1)
        y0 = frame_features[:, :-1].unsqueeze(1).expand(-1, self.max_states, -1, -1)
        y1 = frame_features[:, 1:].unsqueeze(1).expand(-1, self.max_states, -1, -1)
        delta = y1 - y0
        transition = self.transition_mlp(
            torch.cat([current, following_process, y0, y1, delta], dim=-1)
        )
        end = self.end_mlp(torch.cat([current, y0, delta], dim=-1))
        return start, transition, end

    def forward(
        self,
        query_tokens: Tensor,
        video_features: Tensor,
        query_padding_mask: Tensor | None = None,
    ) -> ContentAdaptiveProcessOutput:
        if video_features.ndim != 5:
            raise ValueError("video_features must have shape [batch,frames,channels,H,W]")
        batch, frames, channels, height, width = video_features.shape
        process, language_attention, positions, length_prior, halt = self.compiler(
            query_tokens, query_padding_mask
        )
        spatial = self.spatial_down(video_features.reshape(batch * frames, channels, height, width))
        spatial_height, spatial_width = spatial.shape[-2:]
        spatial = spatial.flatten(2).transpose(1, 2).reshape(batch, frames, -1, spatial.shape[1])
        spatial = self.spatial_norm(spatial)
        spatial_map = spatial.transpose(2, 3).reshape(
            batch, frames, spatial.shape[-1], spatial_height, spatial_width
        )
        query = torch.nn.functional.normalize(self.process_projection(process), dim=-1)
        visual = torch.nn.functional.normalize(self.visual_projection(spatial), dim=-1)
        attention_logits = torch.einsum("bmd,btsd->bmts", query, visual) / math.sqrt(query.shape[-1])
        spatial_attention = attention_logits.softmax(dim=-1)
        event_frame = torch.einsum("bmts,btsd->bmtd", spatial_attention, spatial)
        event_projected = torch.nn.functional.normalize(self.visual_projection(event_frame), dim=-1)
        compatibility = torch.einsum("bmd,bmtd->bmt", query, event_projected)
        frame_features = event_frame.mean(dim=1)
        start_logits, transition_logits, end_logits = self._transition_logits(process, frame_features)
        alignment = self.alignment(
            compatibility, length_prior, start_logits, transition_logits, end_logits
        )
        context = torch.einsum(
            "btm,bmd->btd", alignment.posterior_process, self.context_projection(process)
        )
        adapter_input = torch.cat(
            [
                context,
                alignment.process_occupancy.unsqueeze(-1),
                alignment.transition_confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        frame_states = self.beta * self.adapter(adapter_input)
        return ContentAdaptiveProcessOutput(
            frame_states=frame_states,
            process_states=process,
            language_attention=language_attention,
            language_positions=positions,
            compatibility=compatibility,
            alignment=alignment,
            halt_probabilities=halt,
            length_prior=length_prior,
            beta=self.beta,
            spatial_map=spatial_map,
            frame_features=frame_features,
            transition_mode=self.transition_mode,
        )

    def score_training_object_masks(
        self, output: ContentAdaptiveProcessOutput, object_masks: Tensor
    ) -> Tensor:
        if object_masks.ndim != 5:
            raise ValueError("object_masks must have shape [batch,objects,frames,H,W]")
        batch, objects, frames, height, width = object_masks.shape
        spatial = output.spatial_map
        masks = torch.nn.functional.interpolate(
            object_masks.reshape(batch * objects * frames, 1, height, width).float(),
            size=spatial.shape[-2:], mode="nearest",
        ).reshape(batch, objects, frames, 1, *spatial.shape[-2:])
        denominator = masks.sum(dim=(-1, -2)).clamp_min(1.0)
        trajectory = (spatial.unsqueeze(1) * masks).sum(dim=(-1, -2)) / denominator
        # Object-relative emissions retain the same language states and length prior.
        process = torch.nn.functional.normalize(
            self.process_projection(output.process_states), dim=-1
        )
        object_visual = torch.nn.functional.normalize(self.visual_projection(trajectory), dim=-1)
        emissions = torch.einsum("bmd,bktd->bkmt", process, object_visual)
        scores = []
        for object_index in range(objects):
            object_frame = trajectory[:, object_index]
            start, transition, end = self._transition_logits(output.process_states, object_frame)
            aligned = self.alignment(
                emissions[:, object_index], output.length_prior, start, transition, end
            )
            scores.append(aligned.score)
        return torch.stack(scores, dim=-1)

"""Frame-aware grounding bridge into the frozen SAM3.1 executor."""

from __future__ import annotations

import torch
from torch import nn


class FrameQueryFusion(nn.Module):
    """QwenSeg-SAM31 baseline fusion with no process reasoning."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.GELU(),
        )
        self.gate = nn.Linear(bottleneck_dim, 1)
        self.update = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(
        self, frame_summaries: torch.Tensor, query_global: torch.Tensor
    ) -> torch.Tensor:
        if query_global.ndim == 1:
            query_global = query_global.unsqueeze(0)
        if query_global.shape[0] == 1:
            query_global = query_global.expand(frame_summaries.shape[0], -1)
        joined = self.norm(torch.cat([frame_summaries, query_global], dim=-1))
        condition = self.condition(joined)
        return frame_summaries + torch.sigmoid(self.gate(condition)) * self.update(condition)


class QwenToSAM31Bridge(nn.Module):
    """Representation translator; it never predicts masks or pixel logits."""

    def __init__(
        self,
        qwen_dim: int = 2560,
        sam_prompt_dim: int = 256,
        bottleneck_dim: int = 512,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, sam_prompt_dim),
        )

    def forward(self, grounding_states: torch.Tensor) -> torch.Tensor:
        return self.layers(grounding_states)


def multiplex_prompt_batch(
    frame_prompts: torch.Tensor,
    output_valid_embed: torch.Tensor,
    output_invalid_embed: torch.Tensor,
    object_slot: int = 0,
) -> torch.Tensor:
    """Insert one frame prompt per bucket while retaining SAM3.1 validity priors.

    Official SAM3.1 represents a bucket as ``[multiplex_count, 256]`` and adds
    ``extra_per_object_embeddings`` to decoder mask tokens.  QwenProcessSeg uses
    one bucket per prompted frame and one stable object slot; unused slots keep
    the checkpoint's invalid-object embedding.  The bridge output is a residual
    on the checkpoint's valid-object embedding, not a replacement decoder.
    """
    if frame_prompts.ndim != 2:
        raise ValueError("frame_prompts must be [T,C]")
    if output_valid_embed.shape != output_invalid_embed.shape:
        raise ValueError("SAM3.1 valid/invalid embeddings must have identical shapes")
    multiplex_count, prompt_dim = output_valid_embed.shape
    if frame_prompts.shape[1] != prompt_dim:
        raise ValueError("bridge output dimension does not match SAM3.1")
    if not 0 <= object_slot < multiplex_count:
        raise ValueError("object_slot outside multiplex bucket")
    batch = output_invalid_embed.unsqueeze(0).expand(
        frame_prompts.shape[0], -1, -1
    ).clone()
    batch[:, object_slot] = output_valid_embed[object_slot] + frame_prompts
    return batch

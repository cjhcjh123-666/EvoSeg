"""Deterministic 16-frame Qwen3-VL protocol and token-span recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class FrameTokenSpan:
    frame_number: int
    source_frame_index: int
    token_start: int
    token_end: int
    grid_thw: tuple[int, int, int]

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


def uniform_frame_indices(num_frames: int, budget: int = 16) -> list[int]:
    """Uniformly cover the complete video without prefix truncation.

    Videos shorter than the budget use all available frames once.  This avoids
    silently duplicating evidence while making the actual count explicit.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    count = min(num_frames, budget)
    if count == 1:
        return [0]
    indices = torch.linspace(0, num_frames - 1, count).round().long().tolist()
    if indices != sorted(set(indices)):
        raise AssertionError(f"uniform sampling produced duplicates: {indices}")
    return indices


def recover_frame_token_spans(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    source_frame_indices: Sequence[int],
    image_token_id: int,
    spatial_merge_size: int,
) -> list[FrameTokenSpan]:
    """Map each independently encoded Qwen frame to its visual-token run.

    Qwen3-VL replaces each ``<|image_pad|>`` position with one merged visual
    token.  Encoding frames as an ordered list of images preserves one token run
    and one ``image_grid_thw`` row per frame, so boundaries are exactly
    recoverable rather than inferred from normalized position.
    """
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("span recovery currently requires batch size 1")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be [L] or [1,L]")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must be [T,3]")
    if len(source_frame_indices) != image_grid_thw.shape[0]:
        raise ValueError("one source frame index is required per grid row")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
    counts = [
        int(grid.prod().item()) // (spatial_merge_size**2)
        for grid in image_grid_thw
    ]
    if sum(counts) != positions.numel():
        raise ValueError(
            f"visual-token mismatch: grids imply {sum(counts)}, "
            f"input contains {positions.numel()}"
        )

    spans: list[FrameTokenSpan] = []
    cursor = 0
    for frame_number, (source_index, count, grid) in enumerate(
        zip(source_frame_indices, counts, image_grid_thw, strict=True)
    ):
        frame_positions = positions[cursor : cursor + count]
        if frame_positions.numel() != count:
            raise AssertionError("incomplete visual-token run")
        start = int(frame_positions[0].item())
        end = int(frame_positions[-1].item()) + 1
        if end - start != count:
            raise ValueError(
                f"frame {frame_number} visual tokens are not contiguous: "
                f"expected {count}, got span [{start},{end})"
            )
        spans.append(
            FrameTokenSpan(
                frame_number=frame_number,
                source_frame_index=int(source_index),
                token_start=start,
                token_end=end,
                grid_thw=tuple(int(value) for value in grid.tolist()),
            )
        )
        cursor += count
    return spans


def gather_frame_tokens(
    hidden_states: torch.Tensor,
    spans: Sequence[FrameTokenSpan],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Return spatial tokens and their per-frame summaries without global pooling."""
    if hidden_states.ndim == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError("frame-token gathering currently requires batch size 1")
        hidden_states = hidden_states[0]
    tokens = [hidden_states[span.token_start : span.token_end] for span in spans]
    if any(frame.numel() == 0 for frame in tokens):
        raise ValueError("empty frame token span")
    summaries = torch.stack([frame.mean(dim=0) for frame in tokens], dim=0)
    return tokens, summaries

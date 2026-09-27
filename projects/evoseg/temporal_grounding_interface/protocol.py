"""Protocol primitives for the dynamic-grounding-interface experiment.

These helpers are deliberately model-free so the causal protocol can be tested
without loading Sa2VA or SAM 3.1.  No helper accepts a ground-truth mask.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


SUPPORTED_STAGE_COUNTS = (1, 2, 4, 8)


def uniform_positions(length: int, count: int) -> list[int]:
    """Return ordered, endpoint-covering positions without prefix truncation."""

    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if length <= count:
        return list(range(length))
    values = [int(round(value)) for value in np.linspace(0, length - 1, count)]
    if len(values) != len(set(values)):
        raise AssertionError("uniform sampling produced duplicate positions")
    return values


def stage_end_positions(frame_count: int, stages: int) -> list[int]:
    """Select nested *segment-end* positions from a canonical eight-stage grid.

    Unlike endpoint-covering frame sampling, a stage boundary must not put the
    first stage at frame zero: stage 1 contains the first eighth of the video.
    The ``ceil(k*n/8)-1`` construction gives disjoint, exhaustive temporal
    segments and makes K=1/2/4 exact subsets of K=8.
    """

    if stages not in SUPPORTED_STAGE_COUNTS:
        raise ValueError(f"unsupported stage count: {stages}")
    if frame_count < 8:
        raise ValueError("the eight-stage protocol requires at least eight frames")
    canonical = [int(np.ceil(frame_count * k / 8.0)) - 1 for k in range(1, 9)]
    if stages == 1:
        return [canonical[-1]]
    stride = 8 // stages
    return [canonical[index] for index in range(stride - 1, 8, stride)]


def cumulative_visible_positions(
    frame_count: int, stage_end: int, maximum_frames: int = 16
) -> list[int]:
    """Uniformly sample the observed prefix, including its first and last frame."""

    if not 0 <= stage_end < frame_count:
        raise ValueError("stage endpoint outside video")
    positions = uniform_positions(stage_end + 1, min(maximum_frames, stage_end + 1))
    assert positions[0] == 0
    assert positions[-1] == stage_end
    return positions


def anchor_only_positions(stage_end: int, token_matched_frames: int = 16) -> list[int]:
    """Static control: repeat the current anchor, exposing no other time point.

    Repetition keeps image count and visual-token rules matched to the temporal
    branch while preventing the control from observing cross-frame history.
    """

    if stage_end < 0 or token_matched_frames <= 0:
        raise ValueError("invalid anchor or frame count")
    return [stage_end] * token_matched_frames


def deterministic_positive_point(mask: np.ndarray) -> tuple[float, float]:
    """Return a relative-coordinate point safely inside a candidate mask.

    The point is computed from the candidate only.  Ground truth is neither an
    argument nor an implicit dependency.
    """

    from scipy.ndimage import distance_transform_edt

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError("candidate mask must be a non-empty 2-D array")
    distance = distance_transform_edt(value)
    row, column = np.unravel_index(int(distance.argmax()), distance.shape)
    height, width = value.shape
    return ((float(column) + 0.5) / width, (float(row) + 0.5) / height)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("mask shapes differ")
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def select_candidate(reference_mask: np.ndarray, candidates: list[np.ndarray]) -> int | None:
    """Select by overlap with frozen Sa2VA grounding; ties keep official order."""

    if not candidates:
        return None
    scores = [mask_iou(reference_mask, candidate) for candidate in candidates]
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def stable_identity(
    dataset: str, video_id: str, object_id: str, expression_id: str
) -> str:
    return "/".join(map(str, (dataset, video_id, object_id, expression_id)))


def frame_signature(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Condition:
    name: str
    state_kind: str
    stages: int

    def __post_init__(self) -> None:
        if self.state_kind not in {"global", "static", "temporal"}:
            raise ValueError(self.state_kind)
        if self.stages not in SUPPORTED_STAGE_COUNTS:
            raise ValueError(self.stages)
        if self.state_kind == "global" and self.stages != 1:
            raise ValueError("global condition must have one update")


CONDITIONS = (
    Condition("single_global", "global", 1),
    Condition("static_update_k2", "static", 2),
    Condition("static_update_k4", "static", 4),
    Condition("static_update_k8", "static", 8),
    Condition("temporal_update_k2", "temporal", 2),
    Condition("temporal_update_k4", "temporal", 4),
    Condition("temporal_update_k8", "temporal", 8),
)


def assert_source_video_disjoint(train_videos: set[str], eval_videos: set[str]) -> None:
    overlap = set(map(str, train_videos)) & set(map(str, eval_videos))
    if overlap:
        raise RuntimeError(f"source-video train/eval leakage: {sorted(overlap)[:5]}")

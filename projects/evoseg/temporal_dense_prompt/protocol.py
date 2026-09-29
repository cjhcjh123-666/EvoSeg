from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

STAGES = (1, 3, 5, 7)


def identity(item: dict, expression: dict) -> str:
    return "/".join((item["dataset"], item["video_id"], str(item["object_id"]), str(expression["expression_id"])))


def stage_endpoints(frame_count: int) -> list[int]:
    return [int(round((stage + 1) * (frame_count - 1) / 8)) for stage in STAGES]


def state_index(records_path: Path) -> dict[str, Path]:
    result = {}
    sources = sorted(records_path.rglob("stage_grounding_records.jsonl")) if records_path.is_dir() else [records_path]
    for source in sources:
      with source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            path = Path(row["state_path"])
            if not path.is_absolute():
                path = source.parent / path
            result[row["identity"]] = path
    return result


def video_fold(video_id: str, fraction: float = 0.2) -> str:
    value = int(hashlib.sha256(("tdsp-fixed-split-v1:" + video_id).encode()).hexdigest()[:8], 16)
    return "internal_val" if value % 10_000 < int(fraction * 10_000) else "train"


class DensePromptHead(nn.Module):
    """Small dot-product spatial conditioner; identical for both controls."""

    def __init__(self, state_dim: int = 256, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.state = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.spatial = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.state(state), dim=-1)
        features = F.normalize(self.spatial(spatial), dim=1)
        return torch.einsum("bchw,bc->bhw", features, query) * math.sqrt(features.shape[1]) + self.bias


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum((-2, -1))
    denominator = probability.sum((-2, -1)) + target.sum((-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def training_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def resize_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    if not path.is_file():
        return torch.zeros(size, dtype=torch.float32)
    with Image.open(path) as image:
        value = torch.from_numpy((np.asarray(image.convert("L")) > 0).astype(np.float32))[None, None]
    return F.interpolate(value, size=size, mode="nearest")[0, 0]


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    intersection = np.logical_and(prediction, target).sum()
    iou = float(intersection / union) if union else float(prediction.sum() == target.sum())
    dice = float(2 * intersection / (prediction.sum() + target.sum())) if prediction.any() or target.any() else 1.0
    return {"IoU": iou, "Dice": dice, "J_and_F": (iou + dice) / 2}


def local_extrema_points(heatmap: np.ndarray, positives: int = 4, negatives: int = 4) -> tuple[list[list[float]], list[int]]:
    """Fixed 3x3 NMS and deterministic coordinate tie-breaking; no GT input."""
    value = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    height, width = value.shape[-2:]
    pooled = F.max_pool2d(value, 3, stride=1, padding=1)
    maxima = ((value == pooled) * value).flatten()
    order = sorted(range(maxima.numel()), key=lambda p: (-float(maxima[p]), p))[:positives]
    points, labels = [], []
    used = set()
    for position in order:
        y, x = divmod(position, width)
        used.add((y, x)); points.append([(x + 0.5) / width, (y + 0.5) / height]); labels.append(1)
    foreground = np.asarray(heatmap) >= 0.5
    ys, xs = np.where(foreground)
    candidates = []
    if len(xs):
        x0, x1 = max(0, int(xs.min()) - 2), min(width - 1, int(xs.max()) + 2)
        y0, y1 = max(0, int(ys.min()) - 2), min(height - 1, int(ys.max()) + 2)
        candidates = [(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    if not candidates:
        candidates = [(y, x) for y in range(height) for x in range(width)]
    candidates = sorted((p for p in candidates if p not in used), key=lambda p: (float(heatmap[p]), p[0], p[1]))[:negatives]
    for y, x in candidates:
        points.append([(x + 0.5) / width, (y + 0.5) / height]); labels.append(0)
    return points, labels


def one_point(heatmap: np.ndarray) -> tuple[list[list[float]], list[int]]:
    height, width = heatmap.shape
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return [[(x + 0.5) / width, (y + 0.5) / height]], [1]


def load_feature(path: Path) -> torch.Tensor:
    value = np.load(path)["feature"]
    return torch.from_numpy(value.astype(np.float32))


def count_parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)

"""Dependency-light J&F metrics for FTG pilot validation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from third_parts.revos.utils.metircs import _seg2bmap, db_eval_iou


def boundary_f(annotation: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    values = []
    for gt_mask, foreground_mask in zip(annotation.astype(bool), segmentation.astype(bool)):
        radius = int(np.ceil(0.008 * np.linalg.norm(foreground_mask.shape)))
        coordinates = np.arange(-radius, radius + 1)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        footprint = ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
        fg_boundary = _seg2bmap(foreground_mask)
        gt_boundary = _seg2bmap(gt_mask)
        fg_dilated = cv2.dilate(fg_boundary.astype(np.uint8), footprint)
        gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), footprint)
        n_fg, n_gt = int(fg_boundary.sum()), int(gt_boundary.sum())
        if n_fg == 0 and n_gt > 0:
            precision, recall = 1.0, 0.0
        elif n_fg > 0 and n_gt == 0:
            precision, recall = 0.0, 1.0
        elif n_fg == 0 and n_gt == 0:
            precision = recall = 1.0
        else:
            precision = float((fg_boundary * gt_dilated).sum()) / n_fg
            recall = float((gt_boundary * fg_dilated).sum()) / n_gt
        values.append(
            0.0 if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    return np.asarray(values)


def logits_to_masks(logits: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(
        logits[:, None].float(),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return (resized.sigmoid() >= 0.5).cpu().numpy()


def evaluate_masks(target: torch.Tensor, prediction: np.ndarray) -> dict[str, float]:
    ground_truth = target.cpu().numpy().astype(bool)
    # The DAVIS-style implementation defines two empty masks as J=1 but performs
    # the division before replacing that entry. Silence only that expected 0/0.
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.asarray(db_eval_iou(ground_truth, prediction), dtype=float)
    f = boundary_f(ground_truth, prediction)
    present = ground_truth.any(axis=(-2, -1))
    absent = ~present
    predicted_present = prediction.any(axis=(-2, -1))
    return {
        "j": float(j.mean()),
        "f": float(f.mean()),
        "jf": float((j.mean() + f.mean()) / 2),
        "present_jf": (
            float((j[present].mean() + f[present].mean()) / 2)
            if present.any() else 0.0
        ),
        "false_accept": (
            float(predicted_present[absent].mean()) if absent.any() else 0.0
        ),
        "false_reject": (
            float((~predicted_present[present]).mean()) if present.any() else 0.0
        ),
    }

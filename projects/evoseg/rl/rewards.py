"""Faithfulness rewards for EvoSeg GRPO (image referring segmentation).

kinds:
  present : query refers to an object that exists -> reward mask quality (IoU)
  absent  : query refers to an object that does NOT exist -> reward abstention
"""
import numpy as np


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred/gt: bool HxW. Returns IoU in [0,1]."""
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / (union + 1e-8)


def reward_present(pred_masks, gt_mask, acc_thresh=0.5):
    """pred_masks: list of bool HxW (usually 0 or 1). gt_mask: bool HxW.
    Returns (r_iou, r_acc, n_masks)."""
    if gt_mask is None or len(pred_masks) == 0:
        return 0.0, 0.0, len(pred_masks)
    # merge all predicted masks (usually one)
    pred = np.zeros_like(gt_mask, dtype=bool)
    for m in pred_masks:
        if m.shape == gt_mask.shape:
            pred |= m
    iou = mask_iou(pred, gt_mask)
    acc = 1.0 if iou >= acc_thresh else 0.0
    return float(iou), float(acc), len(pred_masks)


def reward_absent(pred_masks, area_thresh=0.1):
    """Empty-target. If no mask emitted -> perfect abstention.
    If a mask IS emitted -> shape penalty on its area so tiny masks get partial
    credit (provides a learning gradient away from hallucination).
    Returns (r_abstain, mask_area, n_masks)."""
    if len(pred_masks) == 0:
        return 1.0, 0.0, 0
    # use the largest predicted mask's relative area
    area = 0.0
    for m in pred_masks:
        a = float(m.mean())
        if a > area:
            area = a
    r = 1.0 - min(area / area_thresh, 1.0)
    return float(r), float(area), len(pred_masks)


def format_reward(pred_text: str, kind: str) -> float:
    """Tiny format prior: present -> expect a [SEG] token; absent -> prefer no [SEG]."""
    if kind == 'present':
        return 1.0 if '[SEG]' in pred_text else 0.0
    else:
        return 0.0 if '[SEG]' in pred_text else 1.0

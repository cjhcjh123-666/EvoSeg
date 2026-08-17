"""Faithfulness rewards for EvoSeg GRPO (image + video referring segmentation).

kinds:
  present : query refers to an object that exists -> reward mask quality (IoU)
  absent  : query refers to an object that does NOT exist -> reward abstention

Additional rewards (v2):
  reward_counterfactual_pair : (V, q+) must be segmented, (V, q-) must be empty,
                               plus an inconsistency penalty for indiscriminate
                               segmentation (mask emitted for BOTH q+ and q-).
  reward_temporal_absence    : pixel-level penalty on mask area during frames
                               where the target is known NOT to exist.
  reward_temporal_presence   : mean IoU over frames where the target exists.
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


def reward_counterfactual_pair(pred_masks_plus, gt_mask, pred_masks_minus,
                               lambda_abstain=0.5, gamma_inconsistency=0.3,
                               acc_thresh=0.5, area_thresh=0.1):
    """Counterfactual pair reward for one image V and two queries on the SAME V.

    q+ (factual):   the referent exists -> reward mask quality (as present).
    q- (counterfactual): the referent does NOT exist -> reward abstention.

    R_pair = R_seg(q+) + lambda * R_abstain(q-) - gamma * R_inconsistency

    R_inconsistency = 1 iff the model emits >=1 mask for BOTH q+ and q- on the
    same image (the "indiscriminate segmentation" failure mode: it treats every
    query as segmentable, which is exactly the 100%-hallucination behaviour
    observed on gRefCOCO val no-target before RL).

    Returns (r_pair_total_conceptual, info_dict). The trainer spreads
    -gamma*R_inconsistency over the group rollouts.
    """
    r_iou, r_acc, n_plus = reward_present(pred_masks_plus, gt_mask, acc_thresh)
    r_seg = 0.7 * r_iou + 0.3 * r_acc
    r_abs, area_minus, n_minus = reward_absent(pred_masks_minus, area_thresh)
    inconsistent = 1.0 if (n_plus > 0 and n_minus > 0) else 0.0
    r_total = r_seg + lambda_abstain * r_abs - gamma_inconsistency * inconsistent
    info = {
        'r_seg': float(r_seg), 'r_abstain': float(r_abs),
        'inconsistent': float(inconsistent), 'iou': float(r_iou),
        'n_plus': int(n_plus), 'n_minus': int(n_minus),
        'area_minus': float(area_minus),
    }
    return float(r_total), info


def reward_temporal_absence(frame_masks, absent_frames, area_thresh=0.1):
    """Pixel-level temporal absence reward.

    frame_masks : list/array of bool HxW per frame (indexed by video frame).
    absent_frames : iterable of frame indices where the target does NOT exist.

    R = -(1/|A|) * sum_{t in A} min(area_t / area_thresh, 1)

    where area_t = mean(|M_t|) over the frame. Every hallucinated mask pixel
    during absence reduces the reward, so the model learns to STOP propagating
    a mask (e.g. after the target disappears, or SAM2 memory hallucination).

    Returns (r, mean_area_absent, n_absent_frames_used).
    """
    if frame_masks is None or absent_frames is None:
        return 0.0, 0.0, 0
    areas = []
    for t in absent_frames:
        if t >= len(frame_masks):
            continue
        m = frame_masks[t]
        if m is None:
            continue
        areas.append(float(np.asarray(m).mean()))
    if not areas:
        return 0.0, 0.0, 0
    r = -float(np.mean([min(a / area_thresh, 1.0) for a in areas]))
    return r, float(np.mean(areas)), len(areas)


def reward_temporal_presence(frame_masks, gt_masks, present_frames, acc_thresh=0.5):
    """Mean IoU over frames where the target exists (must be segmented).

    Returns (mean_iou, n_present_frames_used).
    """
    if frame_masks is None or gt_masks is None or present_frames is None:
        return 0.0, 0
    ious = []
    for t in present_frames:
        if t >= len(frame_masks):
            continue
        m, g = frame_masks[t], gt_masks[t]
        if m is None or g is None:
            continue
        ious.append(mask_iou(np.asarray(m, dtype=bool), np.asarray(g, dtype=bool)))
    if not ious:
        return 0.0, 0
    return float(np.mean(ious)), len(ious)

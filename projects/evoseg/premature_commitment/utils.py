"""Shared utilities for the M5 premature-commitment pilot.

Design notes
------------
* No silent fallbacks: every degraded path prints and records a flag.
* All machine paths are CLI-overridable; the repo's standard artifact root is
  only a default.
* Ref-YT-VOS annotation dirs are named by *expression id* (verified in M5.0):
  expressions sharing the same obj_id carry that object's masks.  The GT for a
  case is therefore read from `Annotations/{video_id}/{exp_id}/`, and the
  distractor instances are reached through other expressions' obj_ids.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# defaults (all CLI-overridable)
# --------------------------------------------------------------------------
DEFAULT_ARTIFACT_ROOT = '/9950backfile/chenjiahui/evo_artifacts'
DEFAULT_RYVOS_ROOT = os.path.join(DEFAULT_ARTIFACT_ROOT, 'datasets', 'ref_youtube_vos')
DEFAULT_VALID_DIR = os.path.join(DEFAULT_RYVOS_ROOT, 'extracted', 'valid')
DEFAULT_JPEG_ROOT = os.path.join(DEFAULT_VALID_DIR, 'JPEGImages')
DEFAULT_ANN_ROOT = os.path.join(DEFAULT_VALID_DIR, 'Annotations')
DEFAULT_META = os.path.join(DEFAULT_VALID_DIR, 'meta_expressions_challenge.json')
DEFAULT_MANIFEST = os.path.join(DEFAULT_RYVOS_ROOT, 'faithfulness_valid.json')
DEFAULT_MODEL = os.path.join(DEFAULT_ARTIFACT_ROOT, 'models', 'EvoSeg-Qwen3-VL-4B-Faithful')

# `num_frames = min(5, len(video))` in modeling_sa2va_qwen.predict_forward ->
# the SAM2 language prompt is applied on the first N_DEFAULT_PROMPT frames and
# then propagated over the whole video.
N_DEFAULT_PROMPT_FRAMES = 5


# --------------------------------------------------------------------------
# metadata / reproducibility
# --------------------------------------------------------------------------
def git_sha(repo_root: str) -> str:
    try:
        out = subprocess.run(['git', '-C', repo_root, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or 'unknown'
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f'[utils] git_sha failed: {exc}', flush=True)
        return 'unknown'


def write_run_metadata(path: str, **kw) -> dict:
    meta = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': ' '.join(kw.pop('command', [])) if isinstance(kw.get('command', None), (list, tuple)) else kw.pop('command', None),
    }
    meta.update(kw)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print(f'[utils] metadata -> {path}', flush=True)
    return meta


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_manifest(path: str) -> List[dict]:
    with open(path) as fh:
        man = json.load(fh)
    cases = man['cases'] if isinstance(man, dict) and 'cases' in man else man
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f'empty/invalid manifest: {path}')
    return cases


def load_meta(meta_path: str) -> Dict[str, dict]:
    with open(meta_path) as fh:
        return json.load(fh)['videos']


def obj_to_exp(meta_video: dict) -> Dict[str, List[str]]:
    """obj_id -> [exp_id, ...] for one video (from the official meta)."""
    out: Dict[str, List[str]] = {}
    for exp_id, e in meta_video['expressions'].items():
        out.setdefault(str(e['obj_id']), []).append(str(exp_id))
    return out


def _read_mask(ann_root: str, vid: str, exp_id: str, frame: str,
               size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    p = os.path.join(ann_root, vid, str(exp_id), f'{frame}.png')
    if not os.path.exists(p):
        return None
    m = np.array(Image.open(p).convert('L')) > 0
    if size is not None and m.shape != (size[1], size[0]):
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    return m


def case_instance_masks(case: dict, ann_root: str, meta: Dict[str, dict],
                        size: Optional[Tuple[int, int]] = None
                        ) -> Tuple[Optional[str], Dict[str, List[np.ndarray]]]:
    """Per-frame GT masks for the target and every distractor instance.

    Returns (target_obj_id, {obj_id: [mask_frame0, ...]}) where a mask is None
    when the instance is not annotated on that frame (== not available).
    """
    vid = case['video_id']
    exp_id = case.get('exp_id')
    if exp_id is None:
        return None, {}
    o2e = obj_to_exp(meta[vid])
    tgt_obj = str(case['obj_id']) if case.get('obj_id') is not None else None
    if tgt_obj is None:
        # fall back to the expression's own object (explicitly recorded)
        for obj, eids in o2e.items():
            if str(exp_id) in eids:
                tgt_obj = obj
                break
    masks: Dict[str, List[np.ndarray]] = {}
    for obj, eids in o2e.items():
        eid = str(exp_id) if tgt_obj == obj and str(exp_id) in eids else eids[0]
        masks[obj] = [_read_mask(ann_root, vid, eid, f, size) for f in case['frames']]
    return tgt_obj, masks


def load_frames(case: dict, jpeg_root: str) -> List[Image.Image]:
    vid = case['video_id']
    return [Image.open(os.path.join(jpeg_root, vid, f + '.jpg')).convert('RGB')
            for f in case['frames']]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        a = np.array(Image.fromarray(a.astype(np.uint8) * 255).resize(
            (b.shape[1], b.shape[0]), Image.NEAREST)) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def per_frame_identity(pred_mask: Optional[np.ndarray], tgt: Optional[np.ndarray],
                       dists: Sequence[Optional[np.ndarray]]) -> dict:
    """IoU of one predicted mask against the target and the best distractor."""
    iou_t = iou(pred_mask, tgt)
    iou_d = max([iou(pred_mask, d) for d in dists], default=0.0)
    return {
        'iou_target': iou_t,
        'iou_distractor_max': iou_d,
        'identity_margin': iou_t - iou_d,
        'id_err': int(iou_d > iou_t),
        'empty': bool(pred_mask is None or not np.asarray(pred_mask).any()),
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """(mean, lo, hi) bootstrap CI over cases; fixed seed for reproducibility."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi)


def case_macro(rows: Sequence[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float('nan')


def frame_micro(rows: Sequence[dict], num_key: str, den_key: str) -> float:
    num = sum(r.get(num_key, 0) for r in rows)
    den = sum(r.get(den_key, 0) for r in rows)
    return float(num / den) if den else float('nan')

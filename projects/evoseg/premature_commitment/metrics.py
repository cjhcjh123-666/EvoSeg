"""M5 metric helpers (identity / commitment / segmentation) with explicit
frame-micro vs case-macro reporting and bootstrap CIs.

Import from the pilot scripts; keep this file free of I/O side effects.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ identity
def identity_metrics(rows: Sequence[dict]) -> Dict[str, float]:
    """rows: per-frame dicts with iou_target / iou_distractor_max / empty.

    Returns frame-micro aggregates plus the identity margin.
    """
    if not rows:
        return {}
    iou_t = np.array([r['iou_target'] for r in rows], dtype=float)
    iou_d = np.array([r['iou_distractor_max'] for r in rows], dtype=float)
    empty = np.array([bool(r['empty']) for r in rows])
    return {
        'n_frames': int(len(rows)),
        'iou_target_mean': float(iou_t.mean()),
        'iou_distractor_mean': float(iou_d.mean()),
        'identity_margin_mean': float((iou_t - iou_d).mean()),
        'id_err_rate': float((iou_d > iou_t).mean()),
        'empty_rate': float(empty.mean()),
    }


def identity_switch_events(per_frame: Sequence[dict]) -> int:
    """Count non-empty runs whose win-instance flips target<->distractor.

    Reliable only when both instances are annotated on the frame, so frames with
    no distractor evidence (iou_d == 0 and iou_t == 0) are ignored.
    """
    win = []
    for r in per_frame:
        if r.get('empty'):
            continue
        if r['iou_target'] == 0 and r['iou_distractor_max'] == 0:
            continue
        win.append('t' if r['identity_margin'] >= 0 else 'd')
    return int(sum(1 for a, b in zip(win, win[1:]) if a != b))


def case_macro(values: Sequence[Optional[float]]) -> Dict[str, float]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {'n': 0, 'mean': float('nan'), 'median': float('nan')}
    return {'n': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v))}


# -------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[Optional[float]], n_resamples: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {'n': 0, 'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {'n': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'lo': float(lo), 'hi': float(hi)}


def paired_delta(before: Sequence[Optional[float]], after: Sequence[Optional[float]],
                 seed: int = 0, n_resamples: int = 1000) -> Dict[str, float]:
    """Per-case paired delta (after - before) with bootstrap CI; pairs dropped
    only when either side is missing (never silently imputed)."""
    pairs = [(b, a) for b, a in zip(before, after) if b is not None and a is not None]
    if not pairs:
        return {'n': 0, 'mean_delta': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    d = np.array([a - b for b, a in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {'n': int(d.size), 'mean_delta': float(d.mean()),
            'median_delta': float(np.median(d)), 'lo': float(lo), 'hi': float(hi)}


# ------------------------------------------------- commitment (needs labels)
def pre_identifiability_hard_rate(cases: Sequence[dict], labels: Dict[str, str],
                                  checkpoint_key: str = 'prefix_last_frame_idx') -> Dict[str, float]:
    """Fraction of human-AMBIGUOUS checkpoints where the model already emits a
    non-empty single-object prediction, and the wrong-identity share.

    `cases` entries need: case_id, checkpoint runs with 'per_frame' and a
    mapping checkpoint->state in `labels` (key: f'{case_id}#{idx}').
    """
    hard = 0; wrong = 0; n = 0
    for c in cases:
        for cp in c.get('checkpoints', []):
            st = labels.get(f"{c['case_id']}#{cp['checkpoint_idx']}")
            if st != 'AMBIGUOUS':
                continue
            n += 1
            if cp.get('n_nonempty_frames', 0) > 0:
                hard += 1
                if cp.get('identity_decision') == 'distractor':
                    wrong += 1
    return {'n_ambiguous_checkpoints': n,
            'pre_id_hard_rate': (hard / n) if n else float('nan'),
            'pre_id_wrong_rate': (wrong / n) if n else float('nan')}

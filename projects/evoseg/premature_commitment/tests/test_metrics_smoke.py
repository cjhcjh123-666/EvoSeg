"""Smoke test for the M5 metric helpers.

No model, dataset or GPU is required: this exercises only the pure-metric
surface used by the M5.3 evaluation scripts, so it runs in any environment
(pytest, or `python tests/test_metrics_smoke.py`).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (bootstrap_ci, identity_metrics,  # noqa: E402
                     identity_switch_events, paired_delta,
                     pre_identifiability_hard_rate)


def _frame(iou_t, iou_d, empty=False):
    return {'iou_target': iou_t, 'iou_distractor_max': iou_d,
            'identity_margin': iou_t - iou_d,
            'id_err': int(iou_d > iou_t), 'empty': empty}


def test_identity_metrics_basic():
    rows = [_frame(0.9, 0.1), _frame(0.2, 0.8), _frame(0.5, 0.5, empty=True)]
    m = identity_metrics(rows)
    assert m['n_frames'] == 3
    assert abs(m['id_err_rate'] - 1 / 3) < 1e-9
    assert abs(m['iou_target_mean'] - (0.9 + 0.2 + 0.5) / 3) < 1e-9
    assert abs(m['empty_rate'] - 1 / 3) < 1e-9


def test_identity_switch_events_ignores_empty_and_blind_frames():
    blind = _frame(0.0, 0.0)
    per_frame = [_frame(0.8, 0.1), blind, _frame(0.1, 0.7), _frame(0.7, 0.2)]
    assert identity_switch_events(per_frame) == 2


def test_bootstrap_ci_is_deterministic():
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    a = bootstrap_ci(vals, n_resamples=1000, seed=0)
    b = bootstrap_ci(vals, n_resamples=1000, seed=0)
    assert a == b, 'bootstrap CI must be reproducible for a fixed seed'
    assert a['n'] == 5
    assert a['lo'] <= a['mean'] <= a['hi']


def test_paired_delta_drops_missing_pairs_only():
    d = paired_delta([0.5, None, 0.2], [0.3, 0.4, 0.1], seed=0, n_resamples=200)
    assert d['n'] == 2
    assert abs(d['mean_delta'] - (-0.15)) < 1e-9


def test_pre_identifiability_rate_needs_human_labels():
    labels = {'v:0#1': 'AMBIGUOUS', 'v:0#2': 'UNIQUE'}
    cases = [{'case_id': 'v:0', 'checkpoints': [
        {'checkpoint_idx': 1, 'n_nonempty_frames': 3,
         'identity_decision': 'distractor'},
        {'checkpoint_idx': 2, 'n_nonempty_frames': 0,
         'identity_decision': 'empty'}]}]
    r = pre_identifiability_hard_rate(cases, labels)
    assert r['n_ambiguous_checkpoints'] == 1
    assert r['pre_id_hard_rate'] == 1.0
    assert r['pre_id_wrong_rate'] == 1.0


if __name__ == '__main__':
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_'):
            _fn()
            print(f'ok {_name}')
    print('SMOKE_OK')

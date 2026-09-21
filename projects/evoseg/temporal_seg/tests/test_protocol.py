import numpy as np

from projects.evoseg.temporal_seg.official_metrics import (
    db_eval_boundary,
    db_eval_iou,
)
from projects.evoseg.temporal_seg.sampling import nested_temporal_indices


def test_nested_sampling_is_full_range_and_nested():
    samples = nested_temporal_indices(101, [8, 16, 32])
    assert samples[8][0] == samples[16][0] == samples[32][0] == 0
    assert samples[8][-1] == samples[16][-1] == samples[32][-1] == 100
    assert set(samples[8]) < set(samples[16]) < set(samples[32])
    assert samples[8] == sorted(samples[8])


def test_official_metrics_identical_and_disjoint():
    gt = np.zeros((2, 20, 30), dtype=np.uint8)
    gt[:, 3:9, 5:12] = 1
    assert np.allclose(db_eval_iou(gt, gt), 1)
    assert np.allclose(db_eval_boundary(gt, gt), 1)
    pred = np.zeros_like(gt)
    pred[:, 12:18, 18:25] = 1
    assert np.allclose(db_eval_iou(gt, pred), 0)
    assert np.allclose(db_eval_boundary(gt, pred), 0)


def test_official_metrics_empty_semantics():
    empty = np.zeros((2, 10, 10), dtype=np.uint8)
    assert np.allclose(db_eval_iou(empty, empty), 1)
    assert np.allclose(db_eval_boundary(empty, empty), 1)

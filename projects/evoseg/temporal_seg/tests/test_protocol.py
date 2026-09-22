import numpy as np

from projects.evoseg.temporal_seg.official_metrics import (
    db_eval_boundary,
    db_eval_iou,
)
from projects.evoseg.temporal_seg.sampling import nested_temporal_indices
from projects.evoseg.temporal_seg.summarize import length_strata, object_type_means


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


def test_description_aggregation_and_length_strata_are_object_weighted():
    records = []
    for expression_id, score, words in (("a", 0.0, 2), ("b", 1.0, 4)):
        records.append(
            {
                "status": "success",
                "video_id": "v1",
                "object_id": "o1",
                "description_type": "static",
                "frame_budget": 16,
                "J": score,
                "F": score,
                "J_and_F": score,
                "description_length_words": words,
                "token_info": {"visual_tokens": 10},
                "peak_memory_bytes": 1,
                "thermal_state": "warm",
                "latency_seconds_synchronized": 1.0,
                "expression_id": expression_id,
            }
        )
    means = object_type_means(records)
    assert means[("v1", "o1", "static", 16)]["J_and_F"] == 0.5
    assert means[("v1", "o1", "static", 16)]["n_expressions"] == 2

    # Add enough already object-aggregated rows to form strata.  The diagnostic
    # counts objects, never the two source expressions above separately.
    for index in range(1, 7):
        means[(f"v{index + 1}", f"o{index + 1}", "static", 16)] = {
            "J_and_F": index / 10,
            "length_words": index + 4,
        }
    strata = length_strata(means, [])
    assert sum(
        row["n_objects"]
        for row in strata["object_type"]
        if row["type"] == "static"
    ) == 7

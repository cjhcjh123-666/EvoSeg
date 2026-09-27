import numpy as np
import pytest

from projects.evoseg.temporal_grounding_interface.protocol import (
    CONDITIONS,
    anchor_only_positions,
    assert_source_video_disjoint,
    cumulative_visible_positions,
    deterministic_positive_point,
    select_candidate,
    stage_end_positions,
)


def test_stage_endpoints_are_nested_and_cover_full_video():
    endpoints = {k: stage_end_positions(235, k) for k in (1, 2, 4, 8)}
    assert endpoints[1][-1] == 234
    assert set(endpoints[1]) < set(endpoints[2]) < set(endpoints[4]) < set(endpoints[8])


def test_cumulative_frames_cover_prefix_and_static_sees_anchor_only():
    visible = cumulative_visible_positions(100, 74, maximum_frames=16)
    assert visible[0] == 0 and visible[-1] == 74
    assert visible == sorted(set(visible))
    assert anchor_only_positions(74, 16) == [74] * 16


def test_candidate_point_is_inside_and_selection_is_gt_free():
    first = np.zeros((20, 20), dtype=bool)
    first[2:7, 2:7] = True
    second = np.zeros_like(first)
    second[12:18, 12:18] = True
    x, y = deterministic_positive_point(second)
    assert second[min(int(y * 20), 19), min(int(x * 20), 19)]
    assert select_candidate(second, [first, second]) == 1


def test_train_eval_overlap_fails_loudly():
    with pytest.raises(RuntimeError, match="leakage"):
        assert_source_video_disjoint({"v1", "v2"}, {"v2", "v3"})


def test_exact_condition_matrix():
    assert [value.name for value in CONDITIONS] == [
        "single_global",
        "static_update_k2",
        "static_update_k4",
        "static_update_k8",
        "temporal_update_k2",
        "temporal_update_k4",
        "temporal_update_k8",
    ]


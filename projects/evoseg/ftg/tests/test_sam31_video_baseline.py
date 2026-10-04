import numpy as np

from projects.evoseg.ftg.evaluate_sam31_video_baseline import (
    _persistent_track_prediction,
    _union_prediction,
)


def test_persistent_track_is_selected_once_from_video_scores():
    outputs = {
        0: {
            "out_obj_ids": np.array([4, 9]),
            "out_binary_masks": np.array([[[1, 0]], [[0, 1]]], dtype=bool),
        },
        1: {
            "out_obj_ids": np.array([9, 4]),
            "out_binary_masks": np.array([[[1, 1]], [[0, 1]]], dtype=bool),
        },
    }
    prediction, chosen = _persistent_track_prediction(
        outputs, {4: 0.6, 9: 0.9}, 2, (1, 2)
    )
    assert chosen == 9
    assert prediction.tolist() == [[[False, True]], [[True, True]]]


def test_no_track_returns_empty_video_and_union_keeps_all_tracks():
    empty, chosen = _persistent_track_prediction({}, {}, 2, (1, 2))
    assert chosen is None
    assert not empty.any()

    outputs = {
        0: {
            "out_obj_ids": np.array([1, 2]),
            "out_binary_masks": np.array([[[1, 0]], [[0, 1]]], dtype=bool),
        }
    }
    union = _union_prediction(outputs, 2, (1, 2))
    assert union.tolist() == [[[True, True]], [[False, False]]]


def test_persistent_selection_ignores_suppressed_unobserved_track():
    outputs = {
        0: {
            "out_obj_ids": np.array([4]),
            "out_binary_masks": np.array([[[1, 0]]], dtype=bool),
        }
    }
    prediction, chosen = _persistent_track_prediction(
        outputs, {4: 0.5, 99: 0.99}, 1, (1, 2)
    )
    assert chosen == 4
    assert prediction.tolist() == [[[True, False]]]

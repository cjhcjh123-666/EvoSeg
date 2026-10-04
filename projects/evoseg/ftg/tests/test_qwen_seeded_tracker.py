import numpy as np
import torch

from projects.evoseg.ftg.evaluate_qwen_seeded_tracker import (
    _bidirectional_track,
    _interior_point,
    _track_mask,
)


def test_interior_point_stays_inside_prediction():
    mask = np.zeros((7, 9), dtype=bool)
    mask[2:6, 3:8] = True
    x, y = _interior_point(mask, torch.zeros(2, 2))
    px = min(int(x * mask.shape[1]), mask.shape[1] - 1)
    py = min(int(y * mask.shape[0]), mask.shape[0] - 1)
    assert mask[py, px]


def test_empty_prediction_falls_back_to_logit_maximum():
    mask = np.zeros((4, 4), dtype=bool)
    logits = torch.tensor([[0.0, 0.0], [0.0, 8.0]])
    x, y = _interior_point(mask, logits)
    assert x > 0.5 and y > 0.5


def test_track_mask_follows_persistent_object_id():
    output = {
        "out_obj_ids": np.array([8, 1]),
        "out_binary_masks": np.array([[[1, 0]], [[0, 1]]], dtype=bool),
    }
    assert _track_mask(output, 1, (1, 2)).tolist() == [[False, True]]
    assert not _track_mask(output, 3, (1, 2)).any()


def test_endpoint_anchor_uses_independent_reverse_state(monkeypatch):
    class Frame:
        height = 1
        width = 2

    class Tracker:
        def __init__(self):
            self.states = 0

        def init_state(self, **_kwargs):
            self.states += 1
            return {"state": self.states}

        def add_mask(self, state, frame_idx, _mask, _object_id):
            mask = np.array([[[state["state"] == 2, True]]], dtype=bool)
            return frame_idx, {"out_obj_ids": np.array([1]), "out_binary_masks": mask}

        def propagate_in_video(self, state, start_frame_idx, reverse, **_kwargs):
            indices = range(start_frame_idx - 1, -1, -1) if reverse else ()
            for index in indices:
                mask = np.array([[[state["state"] == 2, True]]], dtype=bool)
                yield index, {"out_obj_ids": np.array([1]), "out_binary_masks": mask}

    tracker = Tracker()
    monkeypatch.setattr(
        "projects.evoseg.ftg.evaluate_qwen_seeded_tracker._add_predicted_mask",
        lambda tracker, state, frame_idx, mask, object_id: tracker.add_mask(
            state, frame_idx, mask, object_id
        ),
    )
    prediction = _bidirectional_track(
        tracker, [Frame(), Frame()], 1, np.ones((1, 2), dtype=bool)
    )
    assert tracker.states == 2
    assert prediction.tolist() == [[[True, True]], [[True, True]]]

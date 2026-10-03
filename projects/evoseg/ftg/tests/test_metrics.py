import numpy as np
import torch

from projects.evoseg.ftg.metrics import evaluate_masks


def test_perfect_prediction_has_perfect_j_and_f():
    target = torch.zeros(2, 32, 32)
    target[0, 8:20, 9:23] = 1
    prediction = target.numpy().astype(bool)
    metrics = evaluate_masks(target, prediction)
    assert metrics["j"] == 1.0
    assert metrics["f"] == 1.0
    assert metrics["jf"] == 1.0
    assert metrics["false_accept"] == 0.0
    assert metrics["false_reject"] == 0.0


def test_prediction_on_absent_frame_is_false_accept():
    target = torch.zeros(2, 24, 24)
    target[0, 4:12, 5:14] = 1
    prediction = np.zeros((2, 24, 24), dtype=bool)
    prediction[0] = target[0].numpy().astype(bool)
    prediction[1, 2:6, 2:6] = True
    metrics = evaluate_masks(target, prediction)
    assert metrics["false_accept"] == 1.0
    assert metrics["false_reject"] == 0.0
    assert metrics["jf"] < 1.0

import pytest
import torch

from projects.evoseg.qwen_process_seg.baseline_model import query_assignments


def test_predicted_score_uses_oracle_only_for_supervision():
    losses = torch.tensor([[2.0, 0.2, 1.0], [0.1, 2.0, 1.0]])
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.1, 0.8, 0.2]])
    oracle, supervised, predicted = query_assignments(
        losses, scores, "predicted_score"
    )
    assert oracle.tolist() == [1, 0]
    assert supervised.tolist() == [1, 0]
    assert predicted.tolist() == [2, 1]


def test_fixed_slot_uses_the_same_query_for_training_and_inference():
    losses = torch.tensor([[2.0, 0.2, 1.0], [0.1, 2.0, 1.0]])
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.1, 0.8, 0.2]])
    oracle, supervised, predicted = query_assignments(
        losses, scores, "fixed_slot", fixed_query_index=0
    )
    assert oracle.tolist() == [1, 0]
    assert supervised.tolist() == [0, 0]
    assert predicted.tolist() == [0, 0]


def test_consistent_score_uses_one_score_selected_query_for_the_clip():
    losses = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 2.0]])

    oracle, supervised, predicted = query_assignments(
        losses, scores, "consistent_score"
    )

    assert oracle.tolist() == [0, 1, 0]
    assert supervised.tolist() == [0, 1, 0]
    assert predicted.tolist() == [1, 1, 1]


def test_fixed_slot_validates_query_index():
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        query_assignments(values, values, "fixed_slot", fixed_query_index=3)


def test_video_matching_uses_one_query_for_the_entire_clip():
    losses = torch.tensor(
        [
            [0.0, 2.0, 3.0],
            [4.0, 0.0, 3.0],
            [4.0, 0.0, 3.0],
        ]
    )
    scores = torch.zeros_like(losses)

    oracle, supervised, _ = query_assignments(
        losses, scores, "predicted_score", match_scope="video"
    )

    assert oracle.tolist() == [1, 1, 1]
    assert supervised.tolist() == [1, 1, 1]


def test_query_assignment_rejects_unknown_match_scope():
    values = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="unknown match scope"):
        query_assignments(values, values, "predicted_score", match_scope="clipish")

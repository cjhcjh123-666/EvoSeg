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


def test_fixed_slot_validates_query_index():
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        query_assignments(values, values, "fixed_slot", fixed_query_index=3)

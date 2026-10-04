import pytest
import torch

from projects.evoseg.qwen_process_seg.baseline_model import query_selection_loss


def test_binary_objectness_rewards_the_matched_query_and_suppresses_others():
    matched = torch.tensor([1])
    good = torch.tensor([[-4.0, 4.0, -4.0]])
    bad_positive = torch.tensor([[-4.0, -4.0, -4.0]])
    bad_negative = torch.tensor([[4.0, 4.0, 4.0]])

    good_loss = query_selection_loss(good, matched, "binary_objectness")
    assert good_loss < query_selection_loss(
        bad_positive, matched, "binary_objectness"
    )
    assert good_loss < query_selection_loss(
        bad_negative, matched, "binary_objectness"
    )


def test_binary_objectness_has_gradients_for_positive_and_negative_queries():
    scores = torch.zeros(1, 3, requires_grad=True)
    loss = query_selection_loss(
        scores, torch.tensor([1]), "binary_objectness"
    )
    loss.backward()

    assert scores.grad is not None
    assert scores.grad[0, 1] < 0
    assert scores.grad[0, 0] > 0
    assert scores.grad[0, 2] > 0


def test_selection_loss_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown selection loss"):
        query_selection_loss(torch.zeros(1, 2), torch.tensor([0]), "nope")

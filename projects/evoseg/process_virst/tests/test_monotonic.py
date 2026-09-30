import torch

from projects.evoseg.process_virst.monotonic import MonotonicAlignment
from projects.evoseg.process_virst.process_module import ProcessConditioner


def test_correct_order_scores_above_reversed_order():
    align = MonotonicAlignment()
    correct = torch.tensor([[[-4.0, -4.0, 4.0, -4.0, -4.0],
                             [-4.0, -4.0, -4.0, 4.0, -4.0],
                             [-4.0, -4.0, -4.0, -4.0, 4.0]]])
    reversed_frames = correct.flip(-1)
    gate = torch.full((1, 3), 0.999)
    assert align(correct, gate).score.item() > align(reversed_frames, gate).score.item()


def test_same_frame_set_different_order_changes_alignment():
    align = MonotonicAlignment()
    values = torch.tensor([[[3.0, 0.0, -2.0, -4.0], [-3.0, -1.0, 1.0, 4.0]]])
    gate = torch.full((1, 2), 0.95)
    original = align(values, gate)
    permuted = align(values[:, :, torch.tensor([2, 0, 3, 1])], gate)
    assert not torch.allclose(original.score, permuted.score)
    assert not torch.allclose(original.posterior, permuted.posterior)


def test_gradients_are_finite():
    scores = torch.randn(2, 4, 8, requires_grad=True)
    gate_logits = torch.randn(2, 4, requires_grad=True)
    output = MonotonicAlignment()(scores, gate_logits.sigmoid())
    loss = output.score.mean() + output.posterior.square().mean()
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert gate_logits.grad is not None and torch.isfinite(gate_logits.grad).all()


def test_validity_gated_empty_slot_has_negligible_effect():
    align = MonotonicAlignment(eps=1e-8)
    base = torch.tensor([[[4.0, -2.0, -2.0], [-2.0, -2.0, 4.0]]])
    base_gate = torch.ones(1, 2)
    with_empty = torch.cat([base[:, :1], torch.full((1, 1, 3), 100.0), base[:, 1:]], dim=1)
    empty_gate = torch.tensor([[1.0, 0.0, 1.0]])
    score_base = align(base, base_gate).score
    score_empty = align(with_empty, empty_gate).score
    assert torch.allclose(score_base, score_empty, atol=1e-4, rtol=1e-4)


def test_process_conditioner_shapes_and_beta_initialization():
    module = ProcessConditioner(query_dim=32, process_dim=16, heads=4)
    query = torch.randn(2, 7, 32)
    video = torch.randn(2, 6, 256, 32, 32)
    output = module(query, video)
    assert output.frame_states.shape == (2, 6, 16)
    assert output.alignment.shape == (2, 4, 6)
    assert torch.allclose(output.beta, torch.tensor(1.0), atol=1e-6)

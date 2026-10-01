from __future__ import annotations

import inspect

import torch

from projects.evoseg.content_adaptive_process.losses import (
    language_order_loss,
    object_discrimination_loss,
    process_envelope_loss,
    segment_alignment_loss,
)
from projects.evoseg.content_adaptive_process.process_module import (
    ContentAdaptiveProcessConditioner,
)
from projects.evoseg.continuous_process_virst.virst_integration import QueryStateCapture


def test_content_transition_changes_with_visual_delta():
    torch.manual_seed(4)
    module = ContentAdaptiveProcessConditioner(
        query_dim=16, vision_dim=8, process_dim=16, max_states=3
    )
    process = torch.randn(1, 3, 16)
    frames = torch.randn(1, 5, 16)
    _, first, _ = module._transition_logits(process, frames)
    changed = frames.clone()
    changed[:, 3:] += 5.0
    _, second, _ = module._transition_logits(process, changed)
    assert not torch.allclose(first, second)


def test_global_transition_ignores_visual_delta():
    module = ContentAdaptiveProcessConditioner(
        query_dim=16, vision_dim=8, process_dim=16, max_states=3, transition_mode="global"
    )
    process = torch.randn(1, 3, 16)
    first = module._transition_logits(process, torch.randn(1, 5, 16))
    second = module._transition_logits(process, torch.randn(1, 5, 16) * 9)
    assert all(torch.allclose(left, right) for left, right in zip(first, second))


def test_segment_pooling_and_contrastive_loss():
    process = torch.eye(3).unsqueeze(0)
    frames = torch.tensor([[[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])
    gamma = torch.tensor([[[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])
    prior = torch.tensor([[0.0, 0.0, 1.0]])
    good, margin = segment_alignment_loss(process, frames, gamma, prior)
    bad, _ = segment_alignment_loss(process, frames.flip(-1), gamma, prior)
    assert good < bad
    assert margin > 0


def test_language_state_order_loss():
    prior = torch.tensor([[0.0, 0.0, 1.0]])
    ordered = language_order_loss(torch.tensor([[0.1, 0.5, 0.9]]), prior)
    reversed_value = language_order_loss(torch.tensor([[0.9, 0.5, 0.1]]), prior)
    assert ordered < reversed_value


def test_object_discrimination():
    target = torch.tensor([0])
    assert object_discrimination_loss(torch.tensor([[4.0, 0.0]]), target) < object_discrimination_loss(
        torch.tensor([[0.0, 4.0]]), target
    )


def test_envelope_loss():
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    good = process_envelope_loss(torch.tensor([[0.01, 0.99, 0.99, 0.01]]), target)
    bad = process_envelope_loss(torch.tensor([[0.99, 0.01, 0.01, 0.99]]), target)
    assert good < bad


def test_object_mask_scoring_and_segment_shapes():
    torch.manual_seed(2)
    module = ContentAdaptiveProcessConditioner(
        query_dim=16, vision_dim=8, process_dim=16, max_states=3
    )
    output = module(torch.randn(1, 5, 16), torch.randn(1, 6, 8, 16, 16))
    masks = torch.zeros(1, 2, 6, 16, 16)
    masks[:, 0, :, :8, :8] = 1
    masks[:, 1, :, 8:, 8:] = 1
    scores = module.score_training_object_masks(output, masks)
    assert scores.shape == (1, 2)
    scores.sum().backward()
    assert module.process_projection.weight.grad is not None


def test_no_gt_in_inference():
    parameters = inspect.signature(QueryStateCapture._capture_inputs).parameters
    assert {"gt_masks", "object_id", "action_start", "action_end"}.isdisjoint(parameters)

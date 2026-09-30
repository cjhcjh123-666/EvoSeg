from __future__ import annotations

import torch

from projects.evoseg.ordered_process_grounding.model_v2 import (
    DiscriminativeOrderedHead,
    candidate_standardize,
)


def test_candidate_standardize_has_unit_candidate_scale():
    value = torch.tensor([[1.0, 3.0, 5.0]])
    actual = candidate_standardize(value)
    assert torch.allclose(actual.mean(-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(actual.std(-1, unbiased=False), torch.ones(1), atol=1e-5)


def test_single_candidate_is_stable_and_unique():
    assert torch.equal(candidate_standardize(torch.tensor([[4.0]])), torch.zeros(1, 1))


def test_v2_beta_initializes_to_one_and_changes_candidate_ranking():
    model = DiscriminativeOrderedHead(query_dim=8, visual_dim=8, hidden_dim=8, slots=2)
    assert torch.allclose(model.beta_order, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(model.beta_delta, torch.tensor(1.0), atol=1e-6)
    # Fusion math is candidate-relative; unlike v1 it cannot be silenced by alpha=0.01.
    base = candidate_standardize(torch.tensor([[3.0, 2.0]]))
    order = candidate_standardize(torch.tensor([[0.0, 4.0]]))
    delta = candidate_standardize(torch.tensor([[0.0, 2.0]]))
    final = base + model.beta_order * order + model.beta_delta * delta
    assert final.argmax(-1).item() == 1


def test_v2_forward_backpropagates_through_original_and_permutation():
    generator = torch.Generator().manual_seed(42)
    model = DiscriminativeOrderedHead(query_dim=8, visual_dim=8, hidden_dim=8, slots=2)
    query = torch.randn(1, 4, 8, generator=generator)
    tracks = torch.randn(1, 3, 4, 8, generator=generator)
    final, audit = model(torch.randn(1, 3, generator=generator), query, tracks, tracks.flip(2))
    (final.sum() + audit["delta"].sum()).backward()
    assert model.track_projection[1].weight.grad is not None
    assert torch.isfinite(model.track_projection[1].weight.grad).all()

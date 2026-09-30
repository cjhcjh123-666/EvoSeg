from __future__ import annotations

import torch

from projects.evoseg.ordered_process_grounding.model import monotonic_logsumexp
from projects.evoseg.ordered_process_grounding.protocol import (
    block_swap_order,
    fixed_order_negative,
    is_order_sensitive,
    reverse_order,
)


def test_monotonic_alignment_prefers_abc_to_cba():
    # Each row is an event (A,B,C); each column is a frame state.
    abc = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    cba = abc.flip(-1)
    assert monotonic_logsumexp(abc) > monotonic_logsumexp(cba)


def test_same_frames_different_order_changes_ordered_score():
    original = torch.tensor(
        [[5.0, 1.0, 0.0, 0.0], [0.0, 5.0, 1.0, 0.0], [0.0, 0.0, 5.0, 1.0]]
    )
    assert not torch.isclose(monotonic_logsumexp(original), monotonic_logsumexp(original.flip(-1)))


def test_monotonic_alignment_gradient_is_finite():
    scores = torch.randn(2, 4, 8, requires_grad=True)
    monotonic_logsumexp(scores).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_mean_pool_is_permutation_invariant():
    values = torch.Generator().manual_seed(42)
    sequence = torch.randn(2, 8, 16, generator=values)
    assert torch.allclose(sequence.mean(1), sequence[:, reverse_order()].mean(1), atol=1e-7)
    assert torch.allclose(sequence.mean(1), sequence[:, block_swap_order()].mean(1), atol=1e-7)


def test_filter_and_fixed_negative_are_deterministic():
    assert is_order_sensitive("the dog sits, then runs")
    assert not is_order_sensitive("the dog is brown")
    assert fixed_order_negative("x") == fixed_order_negative("x")
    name, order = fixed_order_negative("x")
    assert name in {"reverse", "block_swap"}
    assert sorted(order) == list(range(8))

from __future__ import annotations

import torch
import numpy as np

from projects.evoseg.ordered_process_grounding.model import monotonic_logsumexp
from projects.evoseg.ordered_process_grounding.protocol import (
    block_swap_order,
    fixed_order_negative,
    is_order_sensitive,
    reverse_order,
)
from projects.evoseg.ordered_process_grounding.prepare_features import batch_region_pool
from projects.evoseg.ordered_process_grounding.train_sft import base_logits, order_inputs
from projects.evoseg.ordered_process_grounding.build_groundmore_manifest import (
    official_action_window,
    time_str_to_seconds,
)
from projects.evoseg.temporal_compiler.temporal_matcher_prototype import region_pool


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


def test_batched_region_pool_matches_reference(monkeypatch):
    masks=[np.eye(8,dtype=bool),np.fliplr(np.eye(8,dtype=bool)).copy()]
    tracks=[{"frames":[index]} for index in range(2)]
    monkeypatch.setattr(
        "projects.evoseg.ordered_process_grounding.prepare_features.decode_rle",
        lambda value:masks[value],
    )
    features=torch.randn(16,7)
    actual=batch_region_pool(tracks,[0],[0],{0:features},(4,4))[:,0]
    expected=np.stack([region_pool(features,mask,(4,4))[0].numpy() for mask in masks])
    assert np.allclose(actual,expected,atol=1e-6)


def test_batched_region_pool_retains_zero_candidate_record():
    actual=batch_region_pool([],list(range(8)),list(range(8)),{0:torch.zeros(16,7)},(4,4))
    assert actual.shape==(0,8,7)


def test_frozen_inputs_and_base_logits_are_cached_without_changing_values():
    row={
        "_base_kind":"groundmore",
        "query_tokens":torch.randn(3,5,dtype=torch.float16),
        "tracks":torch.randn(2,8,7,dtype=torch.float16),
    }
    query,tracks=order_inputs(row,"cpu")
    query_again,tracks_again=order_inputs(row,"cpu")
    assert query.data_ptr()==query_again.data_ptr()
    assert tracks.data_ptr()==tracks_again.data_ptr()
    assert torch.allclose(tracks.norm(dim=-1),torch.ones(2,8),atol=1e-5)
    _,reversed_tracks=order_inputs(row,"cpu",reverse_order())
    assert torch.equal(reversed_tracks[0],tracks[0][:,reverse_order()])

    class FrozenBase(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=0
        def forward(self,query_value,track_value):
            self.calls+=1
            return track_value.mean(dim=(1,2)) + query_value.mean()*0

    model=FrozenBase()
    first=base_logits(model,row,"cpu");second=base_logits(model,row,"cpu")
    assert model.calls==1
    assert torch.equal(first,second)


def test_groundmore_action_window_matches_official_six_fps_protocol():
    assert time_str_to_seconds("11:03")==663
    start,end=official_action_window(
        "1eaD62TRpZ0_1103_1113",{"action_start":"11:03","action_end":"11:07"}
    )
    assert (start,end)==(0.0,23.0)

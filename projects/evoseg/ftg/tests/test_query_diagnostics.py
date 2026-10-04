import torch

from projects.evoseg.ftg.train_pilot import _switch_rate


def test_switch_rate_counts_adjacent_identity_slot_changes():
    assert _switch_rate(torch.tensor([2, 2, 4, 4, 1])) == 0.5


def test_switch_rate_is_zero_for_a_single_frame():
    assert _switch_rate(torch.tensor([3])) == 0.0

import numpy as np
import pytest
import torch

from projects.evoseg.dialog.state import SelectionLedger, compose_scoped_mask, mask_recipe
from projects.evoseg.dialog.build_edit_pilot import make_turns


def test_edit_program_and_undo_have_exact_target_supervision():
    registry = {'A': {'anno_ids': [1, 2]}, 'B': {'anno_ids': [3]}}
    turns = make_turns('the person running', 'the person sitting', registry)
    assert [turn['supervision']['active_after'] for turn in turns] == [
        ['A'], ['A', 'B'], ['B'], ['A', 'B'], ['A']]
    assert turns[2]['supervision']['mask_recipe']['anno_ids'] == ['3']
    assert turns[3]['supervision']['mask_recipe']['anno_ids'] == ['1', '2', '3']


def test_invalid_edit_does_not_change_ledger():
    ledger = SelectionLedger(frozenset(['A', 'B']))
    ledger.apply('select', ['A'])
    for operation, targets in [('add', ['A']), ('remove', ['B']), ('select', ['C'])]:
        before = (ledger.active, list(ledger.snapshots))
        with pytest.raises(ValueError):
            ledger.apply(operation, targets)
        assert (ledger.active, ledger.snapshots) == before


def test_scoped_mask_preservation_and_gradients():
    previous = torch.tensor([[0., 1.], [1., 0.]])
    proposal = torch.tensor([[.8, .2], [.4, .9]], requires_grad=True)
    scope = torch.tensor([[0., 1.], [0., 1.]])
    edited = compose_scoped_mask(previous, proposal, scope)
    torch.testing.assert_close(edited[scope == 0], previous[scope == 0], rtol=0, atol=0)
    edited.sum().backward()
    torch.testing.assert_close(proposal.grad, scope)


def test_mask_recipe_preserves_overlap_at_object_level():
    registry = {'A': {'anno_ids': [1]}, 'B': {'anno_ids': [2]}}
    masks = {'1': np.array([[1, 1, 0]], dtype=bool), '2': np.array([[0, 1, 1]], dtype=bool)}
    selected = SelectionLedger(frozenset(registry))
    selected.apply('select', ['A', 'B'])
    selected.apply('remove', ['A'])
    recipe = mask_recipe(selected.active, registry)
    assert recipe == ['2']
    # Removing A must NOT erase overlapping pixels that belong to retained B.
    assert np.array_equal(masks[recipe[0]], np.array([[0, 1, 1]], dtype=bool))

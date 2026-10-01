import inspect

import torch

from projects.evoseg.continuous_process_virst.groundmore_parser import parse_sequential_query
from projects.evoseg.continuous_process_virst.losses import object_discrimination_loss
from projects.evoseg.continuous_process_virst.virst_integration import QueryStateCapture


def test_object_discrimination_loss():
    good = object_discrimination_loss(torch.tensor([[4.0, 0.0]]), torch.tensor([0]))
    bad = object_discrimination_loss(torch.tensor([[0.0, 4.0]]), torch.tensor([0]))
    assert good < bad


def test_groundmore_interval_parser_before():
    value = parse_sequential_query("Who dribbles the ball before scoring a point?")
    assert value.resolved
    assert value.connective == "before"
    assert value.chronological_clauses[0].startswith("Who dribbles")
    assert value.target_clause == value.surface_left


def test_groundmore_interval_parser_after_reorders_process():
    value = parse_sequential_query("Who shoots after dribbling the ball?")
    assert value.resolved
    assert value.chronological_clauses == ("dribbling the ball", "Who shoots")
    assert value.target_clause == "Who shoots"


def test_groundmore_parser_marks_implicit_template_unresolved():
    value = parse_sequential_query("Who pulls the rope causing the baby to fall?")
    assert not value.resolved


def test_no_gt_in_inference_capture_signature():
    parameters = inspect.signature(QueryStateCapture._capture_inputs).parameters
    forbidden = {"gt_masks", "object_id", "action_start", "action_end"}
    assert forbidden.isdisjoint(parameters)

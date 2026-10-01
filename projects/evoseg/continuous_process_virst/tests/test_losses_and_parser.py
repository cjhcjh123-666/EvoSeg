import inspect

import torch
from torch import nn

from projects.evoseg.continuous_process_virst.groundmore_parser import parse_sequential_query
from projects.evoseg.continuous_process_virst.losses import object_discrimination_loss
from projects.evoseg.continuous_process_virst.virst_integration import QueryStateCapture
from projects.evoseg.continuous_process_virst.process_module import ContinuousProcessConditioner
from projects.evoseg.continuous_process_virst.summarize_overfit import summarize


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


def test_training_object_mask_scores_have_candidate_axis():
    torch.manual_seed(3)
    module = ContinuousProcessConditioner(
        query_dim=16,
        vision_dim=8,
        process_dim=16,
        max_states=3,
        heads=4,
    )
    query = torch.randn(1, 5, 16)
    video = torch.randn(1, 4, 8, 16, 16)
    output = module(query, video)
    masks = torch.zeros(1, 2, 4, 16, 16)
    masks[:, 0, :, :8, :8] = 1
    masks[:, 1, :, 8:, 8:] = 1
    scores = module.score_training_object_masks(output, masks)
    assert scores.shape == (1, 2)
    scores.sum().backward()
    assert module.process_projection.weight.grad is not None
    assert torch.isfinite(module.process_projection.weight.grad).all()


def test_overfit_gate_never_passes_without_interval_supervision():
    rows = []
    for step in range(8):
        rows.append(
            {
                "step": step,
                "stage": "joint_sft",
                "question": f"q{step}",
                "segmentation_loss": 2.0 - step * 0.1,
                "verified_order": True,
                "order_margin": 1.0,
                "object_correct": True,
                "expected_process_length": 3.0,
                "mean_state_duration": 2.0,
                "posterior_entropy": 1.0,
            }
        )
    value = summarize(rows, planned=8)
    assert not value["gates"]["interval_localization_improved"]
    assert value["overfit_gate"] == "FAIL"
    assert not value["pilot_authorized"]

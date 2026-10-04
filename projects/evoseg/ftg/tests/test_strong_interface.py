import pytest
import torch

from projects.evoseg.ftg.strong_interface import FactorizedPromptTokens
from projects.sa2va.hf.models_qwen3vl_sam3.ftg_interface import (
    FactorizedPromptTokens as ExportedFactorizedPromptTokens,
)


def test_ftg_keeps_identity_token_fixed_across_frame_states():
    module = FactorizedPromptTokens(hidden_dim=8)
    identity = torch.randn(3, 8)
    first, _ = module(identity, torch.randn(3, 7, 8), variant="ftg")
    second, _ = module(identity, torch.randn(3, 7, 8), variant="ftg")
    assert torch.equal(first[:, 0], identity)
    assert torch.equal(second[:, 0], identity)
    assert not torch.equal(first[:, 1], second[:, 1])


@pytest.mark.parametrize(
    ("variant", "token_count"),
    [
        ("identity_memory", 1),
        ("state_only", 1),
        ("frame_prompt", 1),
        ("vector_sum", 1),
        ("ftg", 2),
        ("anchored_ftg", 1),
    ],
)
def test_variant_shapes_and_gate_range(variant, token_count):
    module = FactorizedPromptTokens(hidden_dim=4)
    tokens, gate = module(
        torch.randn(5, 4), torch.randn(5, 7, 4), variant=variant
    )
    assert tokens.shape == (5, token_count, 4)
    assert gate.shape == (5, 1)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_state_path_receives_gradients_without_changing_identity_values():
    module = FactorizedPromptTokens(hidden_dim=4)
    identity = torch.randn(2, 4, requires_grad=True)
    frame_features = torch.randn(2, 7, 4, requires_grad=True)
    tokens, _ = module(identity, frame_features, variant="ftg")
    tokens[:, 1].square().mean().backward()
    assert identity.grad is not None
    assert frame_features.grad is not None
    assert module.state_mlp[0].weight.grad is not None


def test_anchored_ftg_is_exact_identity_at_initialization_then_learns():
    module = FactorizedPromptTokens(hidden_dim=4)
    identity = torch.randn(2, 4)
    frame_features = torch.randn(2, 7, 4)
    tokens, _ = module(identity, frame_features, variant="anchored_ftg")
    assert torch.equal(tokens[:, 0], identity)

    tokens.square().mean().backward()
    assert module.anchored_state_mlp[-1].weight.grad is not None
    assert module.anchored_state_mlp[-1].weight.grad.abs().sum() > 0


def test_invalid_shapes_and_variant_are_rejected():
    module = FactorizedPromptTokens(hidden_dim=4)
    with pytest.raises(ValueError):
        module(torch.randn(2, 4), torch.randn(2, 3))
    with pytest.raises(ValueError):
        module(torch.randn(2, 4), torch.randn(2, 7, 4), variant="unknown")


def test_exported_hf_composer_is_weight_and_output_compatible():
    training = FactorizedPromptTokens(hidden_dim=8).eval()
    exported = ExportedFactorizedPromptTokens(hidden_dim=8).eval()
    exported.load_state_dict(training.state_dict(), strict=True)
    identity = torch.randn(2, 8)
    frames = torch.randn(2, 5, 8)
    expected, expected_gate = training(identity, frames, variant="ftg")
    actual, actual_gate = exported(identity, frames, variant="ftg")
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity, frames, variant="anchored_ftg"
    )
    actual, actual_gate = exported(
        identity, frames, variant="anchored_ftg"
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

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
        ("unconditioned_residual", 1),
        ("bounded_ftg", 1),
        ("normalized_ftg", 1),
        ("centered_ftg", 1),
        ("gated_centered_ftg", 1),
        ("vector_gated_centered_ftg", 1),
    ],
)
def test_variant_shapes_and_gate_range(variant, token_count):
    module = FactorizedPromptTokens(hidden_dim=4)
    kwargs = (
        {"group_shape": (5, 1)}
        if variant in {
            "centered_ftg", "gated_centered_ftg",
            "vector_gated_centered_ftg",
        }
        else {}
    )
    tokens, gate = module(
        torch.randn(5, 4), torch.randn(5, 7, 4), variant=variant, **kwargs
    )
    assert tokens.shape == (5, token_count, 4)
    expected_gate_width = 4 if variant == "vector_gated_centered_ftg" else 1
    assert gate.shape == (5, expected_gate_width)
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


def test_unconditioned_residual_matches_identity_then_ignores_identity_in_state():
    module = FactorizedPromptTokens(hidden_dim=4)
    identity = torch.randn(2, 4)
    frame_features = torch.randn(2, 7, 4)
    tokens, _ = module(
        identity, frame_features, variant="unconditioned_residual"
    )
    assert torch.equal(tokens[:, 0], identity)

    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(0.1)
    first, first_gate = module(
        identity, frame_features, variant="unconditioned_residual"
    )
    second, second_gate = module(
        identity + 3, frame_features, variant="unconditioned_residual"
    )
    assert torch.allclose(
        first - identity[:, None],
        second - (identity + 3)[:, None],
        atol=1e-6,
    )
    assert torch.equal(first_gate, second_gate)


def test_bounded_ftg_limits_functional_prompt_displacement():
    ratio = 0.02
    module = FactorizedPromptTokens(hidden_dim=4, max_residual_ratio=ratio)
    identity = torch.randn(3, 4)
    frame_features = torch.randn(3, 7, 4)
    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(10)
        module.anchored_state_mlp[-1].bias.fill_(10)
    tokens, _ = module(identity, frame_features, variant="bounded_ftg")
    displacement = (tokens[:, 0] - identity).norm(dim=-1)
    bound = ratio * identity.norm(dim=-1)
    assert torch.all(displacement <= bound + 1e-6)


def test_normalized_ftg_uses_the_allocated_prompt_budget():
    ratio = 0.02
    module = FactorizedPromptTokens(hidden_dim=4, max_residual_ratio=ratio)
    identity = torch.randn(3, 4)
    frame_features = torch.randn(3, 7, 4)
    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(0.1)
        module.anchored_state_mlp[-1].bias.fill_(0.1)
    tokens, _ = module(identity, frame_features, variant="normalized_ftg")
    displacement = (tokens[:, 0] - identity).float().norm(dim=-1)
    budget = ratio * identity.float().norm(dim=-1)
    assert torch.allclose(displacement, budget, atol=1e-5, rtol=1e-4)


def test_centered_ftg_uses_temporal_context_and_exact_budget():
    ratio = 0.02
    module = FactorizedPromptTokens(hidden_dim=4, max_residual_ratio=ratio)
    identity = torch.randn(6, 4)
    frame_features = torch.randn(6, 7, 4)
    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(0.1)
        module.anchored_state_mlp[-1].bias.fill_(0.1)
    tokens, _ = module(
        identity,
        frame_features,
        variant="centered_ftg",
        group_shape=(3, 2),
    )
    displacement = (tokens[:, 0] - identity).float().norm(dim=-1)
    budget = ratio * identity.float().norm(dim=-1)
    assert torch.allclose(displacement, budget, atol=1e-5, rtol=1e-4)

    with pytest.raises(ValueError):
        module(identity, frame_features, variant="centered_ftg")


def test_gated_centered_ftg_has_zero_mean_state_and_gated_budget():
    ratio = 0.02
    module = FactorizedPromptTokens(hidden_dim=4, max_residual_ratio=ratio)
    identity = torch.randn(6, 4)
    frame_features = torch.randn(6, 7, 4)
    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(0.1)
        module.anchored_state_mlp[-1].bias.fill_(0.1)
    state, gate = module.state_observation(
        identity,
        frame_features,
        anchored=True,
        centered=True,
        center_output=True,
        group_shape=(3, 2),
    )
    assert torch.allclose(
        state.reshape(3, 2, 4).mean(dim=0),
        torch.zeros(2, 4),
        atol=1e-6,
    )
    tokens, actual_gate = module(
        identity,
        frame_features,
        variant="gated_centered_ftg",
        group_shape=(3, 2),
    )
    displacement = (tokens[:, 0] - identity).float().norm(dim=-1)
    budget = ratio * identity.float().norm(dim=-1) * gate[:, 0]
    assert torch.equal(gate, actual_gate)
    assert torch.allclose(displacement, budget, atol=1e-5, rtol=1e-4)


def test_vector_gated_centered_ftg_is_channelwise_and_bounded():
    ratio = 0.02
    module = FactorizedPromptTokens(hidden_dim=4, max_residual_ratio=ratio)
    identity = torch.randn(6, 4)
    frame_features = torch.randn(6, 7, 4)
    with torch.no_grad():
        module.anchored_state_mlp[-1].weight.fill_(0.1)
        module.anchored_state_mlp[-1].bias.fill_(0.1)
        module.state_vector_gate[-1].weight.fill_(0.1)
    tokens, gate = module(
        identity,
        frame_features,
        variant="vector_gated_centered_ftg",
        group_shape=(3, 2),
    )
    assert gate.shape == identity.shape
    assert gate.std() > 0
    displacement = (tokens[:, 0] - identity).float().norm(dim=-1)
    bound = ratio * identity.float().norm(dim=-1)
    assert torch.all(displacement <= bound + 1e-6)


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
        identity, frames, variant="bounded_ftg"
    )
    actual, actual_gate = exported(
        identity, frames, variant="bounded_ftg"
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity, frames, variant="normalized_ftg"
    )
    actual, actual_gate = exported(
        identity, frames, variant="normalized_ftg"
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity, frames, variant="centered_ftg", group_shape=(2, 1)
    )
    actual, actual_gate = exported(
        identity, frames, variant="centered_ftg", group_shape=(2, 1)
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity, frames, variant="gated_centered_ftg", group_shape=(2, 1)
    )
    actual, actual_gate = exported(
        identity, frames, variant="gated_centered_ftg", group_shape=(2, 1)
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity,
        frames,
        variant="vector_gated_centered_ftg",
        group_shape=(2, 1),
    )
    actual, actual_gate = exported(
        identity,
        frames,
        variant="vector_gated_centered_ftg",
        group_shape=(2, 1),
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gate, actual_gate)

    expected, expected_gate = training(
        identity, frames, variant="unconditioned_residual"
    )
    actual, actual_gate = exported(
        identity, frames, variant="unconditioned_residual"
    )
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

import pytest
import torch
from torch import nn

from projects.evoseg.ftg.virst_interface import (
    FactorizedVirstSegPrompter,
    IdentityStateComposer,
)
from projects.evoseg.ftg.train_virst_ftg import pad_video_frames_to_multiple


class DummyVirstPrompter(nn.Module):
    token_dim = 8
    max_length = 100
    grid_hw = 8
    nhead = 8

    def __init__(self):
        super().__init__()
        self.register_buffer("prompts", torch.randn(2, 3, 5, 8))
        self.register_buffer("scores", torch.randn(2, 5))

    def forward(self, return_attn=False):
        if return_attn:
            return self.prompts.clone(), self.scores.clone()
        return self.prompts.clone()


def test_factorization_is_exact_and_state_has_zero_temporal_mean():
    prompts = torch.randn(2, 3, 5, 8)
    identity, state = IdentityStateComposer.factorize(prompts)
    assert torch.allclose(identity + state, prompts)
    assert torch.allclose(
        state.mean(dim=2), torch.zeros_like(identity[:, :, 0]), atol=5e-7
    )


@pytest.mark.parametrize("variant", ["ftg", "unconditioned_ftg", "scalar_ftg"])
def test_trainable_ftg_is_exact_public_virst_at_initialization(variant):
    composer = IdentityStateComposer(hidden_dim=8)
    prompts = torch.randn(2, 3, 5, 8)
    output, diagnostics = composer(prompts, variant=variant)
    assert torch.equal(output, prompts)
    assert torch.equal(diagnostics.gate, torch.ones_like(diagnostics.gate))


def test_ftg_training_cannot_change_persistent_identity_mean():
    composer = IdentityStateComposer(hidden_dim=8)
    with torch.no_grad():
        composer.vector_gate[-1].weight.normal_()
        composer.vector_gate[-1].bias.normal_()
    prompts = torch.randn(2, 3, 5, 8)
    expected_identity = prompts.mean(dim=2)
    output, diagnostics = composer(prompts, variant="ftg")
    assert torch.allclose(output.mean(dim=2), expected_identity, atol=1e-6)
    assert diagnostics.gate.min() >= 1 - composer.max_gate_delta
    assert diagnostics.gate.max() <= 1 + composer.max_gate_delta


def test_vector_gate_receives_gradient_from_exact_initialization():
    composer = IdentityStateComposer(hidden_dim=8)
    prompts = torch.randn(2, 3, 5, 8)
    output, _ = composer(prompts, variant="ftg")
    (output * torch.randn_like(output)).sum().backward()
    gradient = composer.vector_gate[-1].weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_identity_and_state_controls_are_structurally_distinct():
    composer = IdentityStateComposer(hidden_dim=8)
    prompts = torch.randn(2, 3, 5, 8)
    identity_only, _ = composer(prompts, variant="identity_only")
    state_only, _ = composer(prompts, variant="state_only")
    assert torch.allclose(
        identity_only,
        prompts.mean(dim=2, keepdim=True).expand_as(prompts),
    )
    assert torch.allclose(
        state_only.mean(dim=2),
        torch.zeros_like(state_only[:, :, 0]),
        atol=5e-7,
    )
    assert not torch.equal(identity_only, state_only)


def test_wrapper_preserves_attention_scores_and_public_output():
    official = DummyVirstPrompter()
    wrapper = FactorizedVirstSegPrompter(official, variant="ftg")
    prompts, scores = wrapper(return_attn=True)
    assert torch.equal(prompts, official.prompts)
    assert torch.equal(scores, official.scores)
    assert wrapper.last_diagnostics is not None


def test_invalid_shape_dimension_and_variant_are_rejected():
    composer = IdentityStateComposer(hidden_dim=8)
    with pytest.raises(ValueError):
        composer(torch.randn(2, 5, 8))
    with pytest.raises(ValueError):
        composer(torch.randn(2, 3, 5, 7))
    with pytest.raises(ValueError):
        composer(torch.randn(2, 3, 5, 8), variant="unknown")


def test_short_video_padding_repeats_last_frame_to_local_group():
    frames = torch.arange(3 * 2).reshape(3, 2)
    padded = pad_video_frames_to_multiple(frames, multiple=4)
    assert padded.shape == (4, 2)
    assert torch.equal(padded[:3], frames)
    assert torch.equal(padded[3], frames[-1])
    aligned = torch.randn(8, 2)
    assert pad_video_frames_to_multiple(aligned, multiple=4) is aligned

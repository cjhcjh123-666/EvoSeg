import torch

from projects.evoseg.ftg.primary_model import EvidenceFactorization


def test_all_main_factorization_branches_receive_gradients():
    torch.manual_seed(7)
    model = EvidenceFactorization(dim=16, heads=4)
    identity = torch.randn(16, requires_grad=True)
    semantic = torch.randn(4, 16, requires_grad=True)
    spatial = torch.randn(4, 9, 16, requires_grad=True)
    prompts, diagnostics = model(identity, semantic, spatial)
    assert prompts.shape == (4, 16)
    assert diagnostics["identity"].shape == (16,)
    assert not torch.equal(prompts[0], prompts[1])
    prompts.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for value in (identity, semantic, spatial):
        assert value.grad is not None and value.grad.abs().sum() > 0


def test_single_frame_and_actual_timestamps_are_supported():
    model = EvidenceFactorization(dim=16, heads=4)
    result, _ = model(torch.randn(16), torch.randn(1, 16), torch.randn(1, 9, 16), torch.zeros(1))
    assert result.shape == (1, 16)
    assert torch.isfinite(result).all()

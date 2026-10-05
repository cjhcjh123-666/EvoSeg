import torch

from projects.evoseg.restart.native_model import native_mask_losses


def test_native_point_losses_have_finite_nonzero_gradients():
    torch.manual_seed(42)
    logits = torch.zeros(2, 8, 8, requires_grad=True)
    targets = torch.zeros(2, 16, 16)
    targets[0, 4:12, 4:12] = 1
    ce, dice = native_mask_losses(logits, targets, num_points=64)
    (ce + dice).backward()
    assert torch.isfinite(ce + dice)
    assert logits.grad.isfinite().all() and logits.grad.abs().sum() > 0


def test_perfect_and_wrong_prediction_point_losses():
    targets = torch.ones(1, 8, 8)
    torch.manual_seed(42)
    correct = sum(native_mask_losses(torch.full((1, 8, 8), 10.), targets, num_points=64))
    torch.manual_seed(42)
    wrong = sum(native_mask_losses(torch.full((1, 8, 8), -10.), targets, num_points=64))
    assert correct < wrong

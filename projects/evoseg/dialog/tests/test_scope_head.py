import torch
import torch.nn.functional as F

from projects.evoseg.dialog.scope_head import TypedScopeHead


def test_typed_scope_predictions_and_gradients_without_gold_inference_inputs():
    head = TypedScopeHead(16, 12, pixel_dim=4, hidden_dim=8)
    output = head(torch.randn(2, 16), torch.randn(2, 3, 12),
                  torch.tensor([[True, False, True], [False, False, False]]), torch.randn(2, 4, 5, 6))
    assert output['operation_logits'].shape == (2, 6)
    assert output['role_pointer_logits'].shape == (2, 2, 4)
    assert torch.isneginf(output['role_pointer_logits'][0, :, 2]).all()
    assert output['role_pointer_logits'][1].softmax(-1)[:, 0].eq(1).all()
    loss = (output['operation_logits'].square().mean() + output['scope'].mean() +
            F.cross_entropy(output['role_pointer_logits'][0], torch.tensor([1, 3])))
    loss.backward()
    assert head.query[1].weight.grad.isfinite().all()
    assert head.pixel_key.weight.grad.abs().sum() > 0
    assert head.role_queries.weight.grad.abs().sum() > 0


def test_first_turn_without_any_history_still_has_null_pointer():
    head = TypedScopeHead(16, 12, pixel_dim=4, hidden_dim=8)
    output = head(torch.randn(1, 16), torch.empty(1, 0, 12),
                  torch.empty(1, 0, dtype=torch.bool), torch.randn(1, 4, 3, 3))
    assert output['role_pointer_logits'].shape == (1, 2, 1)
    assert output['role_pointer_logits'].isfinite().all()

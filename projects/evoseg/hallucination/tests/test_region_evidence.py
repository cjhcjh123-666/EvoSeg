import inspect
from types import SimpleNamespace

import pytest
import torch

from projects.evoseg.hallucination.region_evidence import RegionEvidenceAdapter, pool_regions, predict_regions
from projects.evoseg.hallucination.region_pilot import cached_prediction


def test_pooling_uses_native_threshold_and_raster_order():
    tokens = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    mask = torch.tensor([[[[.6, .4], [.1, -.1]]]])
    foreground, background, global_feature, area = pool_regions(tokens, mask, (2, 2))
    torch.testing.assert_close(foreground, tokens[:1])
    torch.testing.assert_close(background, torch.tensor([[3., 0.]]))
    torch.testing.assert_close(global_feature, torch.tensor([[2.5, 0.]]))
    assert area.item() == .25
    with pytest.raises(ValueError, match='grid'):
        pool_regions(tokens, mask, (1, 3))


def test_region_residual_initially_preserves_prompt_and_has_gradients():
    torch.manual_seed(42)
    head = RegionEvidenceAdapter(8, prompt_dim=4, projection_dim=4)
    prompt = torch.randn(1, 4)
    tensors = [torch.randn(1, 8) for _ in range(5)]
    output = head(prompt, *tensors, torch.tensor([[.25]]))
    torch.testing.assert_close(output['grounding_prompt'], prompt, atol=0., rtol=0.)
    assert output['support_logit'].sigmoid().item() > .5
    loss = output['grounding_prompt'].square().mean() + output['support_logit'].square().mean()
    loss.backward()
    assert head.residual.weight.grad.abs().sum() > 0
    assert head.support.weight.grad.abs().sum() > 0


def test_global_control_cannot_use_candidate_region_or_mask_extent():
    head = RegionEvidenceAdapter(8, prompt_dim=4, projection_dim=4)
    torch.nn.init.normal_(head.residual.weight)
    torch.nn.init.normal_(head.support.weight)
    prompt = torch.randn(1, 4)
    semantic, interaction, global_feature = [torch.randn(1, 8) for _ in range(3)]
    a = head(prompt, semantic, interaction, torch.randn(1, 8), torch.randn(1, 8), global_feature,
             torch.tensor([[.01]]), mode='global')
    b = head(prompt, semantic, interaction, torch.randn(1, 8), torch.randn(1, 8), global_feature,
             torch.tensor([[.99]]), mode='global')
    torch.testing.assert_close(a['grounding_prompt'], b['grounding_prompt'])
    torch.testing.assert_close(a['support_logit'], b['support_logit'])
    assert list(inspect.signature(predict_regions).parameters) == ['runtime', 'image_path', 'query', 'head', 'mode']


def test_empty_region_does_not_produce_nan():
    values = pool_regions(torch.randn(4, 8), torch.zeros(1, 1, 2, 2), (2, 2))
    assert all(torch.isfinite(v).all() for v in values)


def test_private_cached_inference_has_no_gt_or_presence_label_input():
    assert list(inspect.signature(cached_prediction).parameters) == [
        'proposals', 'shape', 'head', 'codec', 'states', 'mode', 'empty_threshold']


def test_cached_prediction_never_consumes_human_support_target():
    head = RegionEvidenceAdapter(8, prompt_dim=4, projection_dim=4)
    proposal = {'prompt': torch.randn(1, 4), 'area': torch.tensor([[.5]]),
                'parent_empty_logit': torch.tensor([-4.]), 'human_target_precision': torch.tensor([0.])}
    proposal.update({key: torch.randn(1, 8) for key in
                    ('semantic', 'interaction', 'foreground', 'background', 'global_feature')})
    codec = SimpleNamespace(model=SimpleNamespace(
        inject_language_embd=lambda *args, **kwargs: torch.ones(1, 1, 2, 2)))
    first = cached_prediction([proposal], (2, 2), head, codec, {}, 'regional')
    proposal['human_target_precision'] = torch.tensor([1.])
    second = cached_prediction([proposal], (2, 2), head, codec, {}, 'regional')
    assert (first == second).all() and first.all()

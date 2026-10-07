import inspect
import json

import pytest
import torch

from projects.evoseg.hallucination.evidence_routing import EvidenceRoutingAdapter, cached_prediction


def values():
    return torch.randn(1, 4, 3, 5), torch.randn(1, 8), torch.tensor([True]), torch.randn(1, 6), torch.randn(1, 1, 6, 10)


def test_initial_prompt_and_native_presence_are_exactly_preserved():
    head = EvidenceRoutingAdapter(8, 4, 4, 6)
    image, query, accepted, prompt, candidate = values()
    output = head(image, query, accepted, prompt, candidate)
    torch.testing.assert_close(output['grounding_prompt'], prompt, atol=0., rtol=0.)
    assert output['existence_logit'].item() == pytest.approx(.1)
    empty = head(image, query, torch.tensor([False]))
    assert empty['existence_logit'].item() == pytest.approx(-.1)
    assert empty['quality_logit'] is None


def test_candidate_changes_never_change_object_existence():
    head = EvidenceRoutingAdapter(8, 4, 4, 6)
    torch.nn.init.normal_(head.existence.weight)
    image, query, accepted, prompt, candidate = values()
    a = head(image, query, accepted, prompt, candidate)
    b = head(image, query, accepted, prompt * 9, candidate * -3)
    c = head(image, query, accepted)
    torch.testing.assert_close(a['existence_logit'], b['existence_logit'], atol=0., rtol=0.)
    torch.testing.assert_close(a['existence_logit'], c['existence_logit'], atol=0., rtol=0.)


def test_low_candidate_quality_is_not_rejection_and_empty_output_can_recover():
    head = EvidenceRoutingAdapter(8, 4, 4, 6)
    image, query, accepted, prompt, candidate = values()
    head.quality.bias.data.fill_(-20)
    output = head(image, query, accepted, prompt, candidate)
    assert output['quality_logit'].sigmoid().item() < .01
    assert output['existence_logit'].sigmoid().item() > .5
    head.existence.bias.data.fill_(5)
    recovered = head(image, query, torch.tensor([False]))
    assert recovered['existence_logit'].sigmoid().item() > .5


def test_gradients_reach_spatial_alignment_existence_correction_and_fallback():
    head = EvidenceRoutingAdapter(8, 4, 4, 6)
    image, query, accepted, prompt, candidate = values()
    a = head(image, query, accepted, prompt, candidate)
    b = head(image, query, torch.tensor([False]))
    loss = a['alignment_logits'].square().mean() + a['existence_logit'].square().mean()
    loss += a['grounding_prompt'].square().mean() + (b['grounding_prompt'] - 1).square().mean()
    loss += a['quality_logit'].square().mean()
    loss.backward()
    for layer in (head.image, head.existence, head.correction, head.fallback, head.quality):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_inference_signature_has_no_ground_truth_and_global_control_is_spatially_constant():
    assert list(inspect.signature(cached_prediction).parameters) == ['head', 'codec', 'states', 'inputs', 'shape', 'mode']
    head = EvidenceRoutingAdapter(8, 4, 4, 6)
    output = head(*values(), mode='global')
    logits = output['alignment_logits']
    torch.testing.assert_close(logits, logits[..., :1, :1].expand_as(logits))
    with pytest.raises(ValueError, match='together'):
        image, query, accepted, prompt, _ = values()
        head(image, query, accepted, prompt=prompt)


def test_legacy_fixed_prior_checkpoint_is_explicitly_preserved(tmp_path):
    legacy = EvidenceRoutingAdapter(8)
    state = legacy.state_dict()
    state.pop('native_prior_log_weight')
    torch.save(state, tmp_path / 'EVIDENCE.pth')
    (tmp_path / 'CONFIG.json').write_text(json.dumps({'language_dim': 8}))
    loaded = EvidenceRoutingAdapter.from_checkpoint(tmp_path)
    result = loaded(torch.randn(1, 256, 2, 2), torch.randn(1, 8), torch.tensor([False]))
    assert result['existence_logit'].item() == pytest.approx(-4.)
    assert not loaded.native_prior_log_weight.requires_grad

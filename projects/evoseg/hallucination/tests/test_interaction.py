import pytest
import torch

from projects.evoseg.hallucination.interaction import InteractionGroundingAdapter, factorial_interaction


def test_additive_priors_cancel_but_interaction_remains():
    torch.manual_seed(42)
    image, query, baseline, true_joint = [torch.randn(2, 8) for _ in range(4)]
    extracted = factorial_interaction(image + query + baseline + true_joint,
                                     image + baseline, query + baseline, baseline)
    torch.testing.assert_close(extracted, true_joint)
    with pytest.raises(ValueError):
        factorial_interaction(image, query[:, :1], baseline, baseline)


def test_untrained_prompt_is_identical_and_trainable():
    torch.manual_seed(42)
    head = InteractionGroundingAdapter(8, prompt_dim=4)
    prompt = torch.randn(2, 4)
    features = [torch.randn(2, 8) for _ in range(4)]
    output = head(prompt, *features)
    torch.testing.assert_close(output['grounding_prompt'], prompt, atol=0., rtol=0.)
    target = torch.randn_like(prompt)
    loss = (output['grounding_prompt'] - target).square().mean() + output['empty_logit'].square().mean()
    loss.backward()
    assert head.prompt_residual.weight.grad.abs().sum() > 0
    assert head.empty_head.weight.grad.abs().sum() > 0

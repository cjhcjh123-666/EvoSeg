from __future__ import annotations

import itertools

import torch

from projects.evoseg.content_adaptive_process.segmental import (
    AdaptiveSegmentalForwardBackward,
    stick_breaking_probabilities,
)


def valid_path(path: tuple[int, ...], length: int) -> bool:
    # 0=PRE, 1..length=process, length+1=POST.
    if any(state < 0 or state > length + 1 for state in path):
        return False
    if 1 not in path or length not in path:
        return False
    for left, right in zip(path, path[1:]):
        if left == 0 and right not in {0, 1}:
            return False
        if 1 <= left < length and right not in {left, left + 1}:
            return False
        if left == length and right not in {length, length + 1}:
            return False
        if left == length + 1 and right != length + 1:
            return False
    return path[-1] in {length, length + 1}


def brute_score(emission, start, transition, end, length):
    frames = emission.shape[-1]
    start_lp = torch.stack([torch.nn.functional.logsigmoid(-start), torch.nn.functional.logsigmoid(start)], -1)
    trans_lp = transition.log_softmax(-1)
    end_lp = end.log_softmax(-1)
    values = []
    for path in itertools.product(range(length + 2), repeat=frames):
        if not valid_path(path, length) or path[0] not in {0, 1}:
            continue
        value = start_lp[0, int(path[0] == 1)]
        if path[0] == 1:
            value = value + emission[0, 0]
        for time in range(1, frames):
            left, right = path[time - 1], path[time]
            if left == 0:
                value = value + start_lp[time, int(right == 1)]
            elif left == length:
                value = value + end_lp[length - 1, time - 1, int(right == length + 1)]
            elif left == length + 1:
                value = value + 0.0
            else:
                value = value + trans_lp[left - 1, time - 1, int(right == left + 1)]
            if 1 <= right <= length:
                value = value + emission[right - 1, time]
        values.append(value)
    return torch.logsumexp(torch.stack(values), dim=0)


def sample_inputs(batch=1, states=3, frames=6):
    torch.manual_seed(9)
    emission = torch.randn(batch, states, frames)
    halt = torch.randn(batch, states)
    prior, _ = stick_breaking_probabilities(halt)
    start = torch.randn(batch, frames)
    transition = torch.randn(batch, states, frames - 1, 2)
    end = torch.randn(batch, states, frames - 1, 2)
    return emission, prior, start, transition, end


def test_stick_breaking_probs_sum_one():
    probability, halt = stick_breaking_probabilities(torch.randn(4, 6))
    assert torch.allclose(probability.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.all(halt[:, -1] == 1)


def test_prefix_consistent_length():
    logits = torch.tensor([[20.0, -20.0, -20.0]])
    probability, _ = stick_breaking_probabilities(logits)
    assert probability[0, 0] > 0.999
    logits = torch.tensor([[-20.0, 20.0, -20.0]])
    probability, _ = stick_breaking_probabilities(logits)
    assert probability[0, 1] > 0.999


def test_forward_partition_vs_bruteforce():
    emission, prior, start, transition, end = sample_inputs(states=2, frames=4)
    output = AdaptiveSegmentalForwardBackward(2)(emission, prior, start, transition, end)
    brute = torch.stack(
        [brute_score(emission[0], start[0], transition[0], end[0], n) for n in (1, 2)]
    )
    assert torch.allclose(output.log_z_by_length[0], brute, atol=1e-5)
    expected = torch.logsumexp(prior[0].log() + brute, dim=0)
    assert torch.allclose(output.score[0], expected, atol=1e-5)


def test_backward_posterior_normalization():
    values = sample_inputs(batch=2)
    output = AdaptiveSegmentalForwardBackward(3)(*values)
    assert torch.allclose(output.posterior_all.sum(-1), torch.ones(2, 6), atol=2e-5)
    assert torch.allclose(output.length_posterior.sum(-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.start_posterior.sum(-1), torch.ones(2), atol=2e-5)
    assert torch.allclose(output.end_posterior.sum(-1), torch.ones(2), atol=2e-5)


def test_pre_process_post_path_validity():
    values = sample_inputs(states=2, frames=5)
    output = AdaptiveSegmentalForwardBackward(2)(*values)
    assert torch.allclose(
        output.posterior_pre + output.process_occupancy + output.posterior_post,
        torch.ones_like(output.process_occupancy), atol=2e-5,
    )


def test_no_backward_transition_and_no_skip_transition():
    emission = torch.tensor([[[8.0, 8.0, -8.0, -8.0], [-8.0, -8.0, 8.0, 8.0]]])
    prior = torch.tensor([[0.0, 1.0]])
    start = torch.full((1, 4), 10.0)
    transition = torch.zeros(1, 2, 3, 2)
    end = torch.zeros(1, 2, 3, 2)
    module = AdaptiveSegmentalForwardBackward(2)
    legal = module(emission, prior, start, transition, end).score
    reverse = module(emission.flip(-1), prior, start, transition, end).score
    assert legal > reverse
    posterior = module(emission, prior, start, transition, end).posterior_process
    assert not torch.any((posterior[:, 1:, 0] > 0.99) & (posterior[:, :-1, 1] > 0.99))


def test_process_can_start_late_and_end_early():
    frames = 7
    emission = torch.full((1, 1, frames), -6.0)
    emission[:, :, 2:5] = 6.0
    prior = torch.ones(1, 1)
    start = torch.tensor([[-8.0, -8.0, 8.0, -8.0, -8.0, -8.0, -8.0]])
    transition = torch.empty(1, 1, frames - 1, 2)
    end = torch.zeros(1, 1, frames - 1, 2)
    end[..., 0] = 3.0
    end[:, :, 4, 1] = 8.0
    output = AdaptiveSegmentalForwardBackward(1)(emission, prior, start, transition, end)
    assert output.posterior_pre[0, 0] > 0.9
    assert output.process_occupancy[0, 3] > 0.9
    assert output.posterior_post[0, -1] > 0.9


def test_gradients_finite():
    values = list(sample_inputs(batch=2))
    for value in (values[0], values[2], values[3], values[4]):
        value.requires_grad_(True)
    output = AdaptiveSegmentalForwardBackward(3)(*values)
    (output.score.mean() + output.posterior_all.square().mean()).backward()
    for value in (values[0], values[2], values[3], values[4]):
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_permutation_changes_sequence_score():
    values = list(sample_inputs(states=2, frames=5))
    module = AdaptiveSegmentalForwardBackward(2)
    original = module(*values).score
    values[0] = values[0].flip(-1)
    permuted = module(*values).score
    assert not torch.allclose(original, permuted)


def test_single_state_edge_case():
    emission, prior, start, transition, end = sample_inputs(states=1, frames=4)
    output = AdaptiveSegmentalForwardBackward(1)(emission, prior, start, transition, end)
    assert output.length_prior.item() == 1.0
    assert torch.isfinite(output.score).all()

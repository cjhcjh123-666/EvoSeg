from __future__ import annotations

import itertools

import torch

from projects.evoseg.continuous_process_virst.segmental import SegmentalForwardBackward


def brute_log_z(emissions, stay, advance, terminal):
    states, frames = emissions.shape
    scores = []
    for changes in itertools.combinations(range(1, frames), terminal):
        changes = set(changes)
        state = 0
        score = emissions[state, 0]
        for time in range(1, frames):
            if time in changes:
                score = score + advance[state]
                state += 1
            else:
                score = score + stay[state]
            score = score + emissions[state, time]
        assert state == terminal
        scores.append(score)
    return torch.logsumexp(torch.stack(scores), dim=0)


def test_segmental_forward_vs_bruteforce():
    torch.manual_seed(1)
    module = SegmentalForwardBackward(3)
    module.stay.data.copy_(torch.tensor([0.1, -0.2, 0.3]))
    module.advance.data.copy_(torch.tensor([0.4, -0.1]))
    emissions = torch.randn(1, 3, 5)
    output = module(emissions, torch.tensor([[0.2, 0.1, -0.3]]))
    expected = torch.stack(
        [brute_log_z(emissions[0], module.stay, module.advance, n) for n in range(3)]
    )
    assert torch.allclose(output.log_z_by_terminal[0], expected, atol=1e-5)


def test_variable_length_marginalization_vs_bruteforce():
    torch.manual_seed(7)
    module = SegmentalForwardBackward(3)
    module.stay.data.copy_(torch.tensor([0.3, -0.4, 0.2]))
    module.advance.data.copy_(torch.tensor([-0.1, 0.5]))
    emissions = torch.randn(1, 3, 5)
    termination_logits = torch.tensor([[0.6, -0.2, 0.1]])
    output = module(emissions, termination_logits)
    brute_terminals = torch.stack(
        [brute_log_z(emissions[0], module.stay, module.advance, n) for n in range(3)]
    )
    expected = torch.logsumexp(
        termination_logits[0].log_softmax(-1) + brute_terminals,
        dim=0,
    )
    assert torch.allclose(output.score[0], expected, atol=1e-5)


def test_forward_backward_normalization():
    output = SegmentalForwardBackward(4)(torch.randn(2, 4, 7), torch.randn(2, 4))
    assert torch.allclose(output.posterior.sum(dim=-1), torch.ones(2, 7), atol=1e-5)
    assert torch.allclose(output.terminal_posterior.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_no_backward_transition_and_permutation_changes_score():
    module = SegmentalForwardBackward(2)
    logits = torch.tensor([[-20.0, 20.0]])
    legal = torch.tensor([[[8.0, 8.0, -8.0, -8.0], [-8.0, -8.0, 8.0, 8.0]]])
    reversed_order = legal.flip(-1)
    assert module(legal, logits).score.item() > module(reversed_order, logits).score.item()


def test_self_loop_duration():
    module = SegmentalForwardBackward(2)
    emissions = torch.tensor([[[8.0, 8.0, 8.0, -8.0, -8.0, -8.0], [-8.0, -8.0, -8.0, 8.0, 8.0, 8.0]]])
    output = module(emissions, torch.tensor([[-20.0, 20.0]]))
    assert output.expected_duration[0, 0] > 2.9
    assert output.expected_duration[0, 1] > 2.9


def test_variable_length_terminal():
    module = SegmentalForwardBackward(3)
    emissions = torch.zeros(1, 3, 5)
    short = module(emissions, torch.tensor([[20.0, -20.0, -20.0]]))
    long = module(emissions, torch.tensor([[-20.0, -20.0, 20.0]]))
    assert short.expected_length.item() < 1.01
    assert long.expected_length.item() > 2.99
    assert long.terminal_posterior.argmax(-1).item() == 2


def test_gradient_finite():
    module = SegmentalForwardBackward(3)
    emissions = torch.randn(2, 3, 6, requires_grad=True)
    termination = torch.randn(2, 3, requires_grad=True)
    output = module(emissions, termination)
    (output.score.mean() + output.posterior.square().mean()).backward()
    assert emissions.grad is not None and torch.isfinite(emissions.grad).all()
    assert termination.grad is not None and torch.isfinite(termination.grad).all()
    assert module.stay.grad is not None and torch.isfinite(module.stay.grad).all()


def test_single_state_edge_case():
    module = SegmentalForwardBackward(1)
    emissions = torch.randn(2, 1, 4)
    output = module(emissions, torch.zeros(2, 1))
    assert torch.allclose(output.posterior, torch.ones(2, 4, 1))
    assert torch.allclose(output.score, emissions.sum(dim=(1, 2)))


def test_m_greater_than_t_error():
    module = SegmentalForwardBackward(4)
    try:
        module(torch.randn(1, 4, 3), torch.randn(1, 4))
    except ValueError as error:
        assert "exceeds frames" in str(error)
    else:
        raise AssertionError("M > T was accepted")

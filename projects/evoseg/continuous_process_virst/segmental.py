"""Log-space continuous segmental alignment with variable terminal state.

Paths start in state 1, end in a learned terminal state N, and may only stay in
the current state or advance by exactly one state at each frame. Consequently
every state 1..N is visited at least once and can persist for multiple frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SegmentalAlignmentOutput:
    score: Tensor
    log_z_by_terminal: Tensor
    terminal_prior: Tensor
    terminal_posterior: Tensor
    posterior: Tensor
    expected_length: Tensor
    expected_duration: Tensor
    posterior_entropy: Tensor


class SegmentalForwardBackward(nn.Module):
    """Differentiable forward-backward for self-loop/forward-one paths."""

    def __init__(self, max_states: int = 6) -> None:
        super().__init__()
        if max_states < 1:
            raise ValueError("max_states must be positive")
        self.max_states = max_states
        self.stay = nn.Parameter(torch.zeros(max_states))
        self.advance = nn.Parameter(torch.zeros(max_states - 1))

    @staticmethod
    def _safe_logsumexp(values: list[Tensor]) -> Tensor:
        """Log-sum-exp whose gradient is finite when every path is unreachable.

        PyTorch correctly returns ``-inf`` for ``logsumexp([-inf, -inf])``, but
        its backward contains an undefined ``0 / 0``.  Segmental DPs encounter
        this case for states that cannot yet have been reached.  Replacing only
        the reduction inputs by the smallest finite value keeps the reduction
        differentiable, while the final ``where`` restores exact ``-inf`` and
        gives unreachable branches zero gradient.
        """

        stacked = torch.stack(values, dim=-1)
        reachable = torch.isfinite(stacked).any(dim=-1)
        finite_floor = torch.finfo(stacked.dtype).min
        safe = stacked.masked_fill(~torch.isfinite(stacked), finite_floor)
        reduced = torch.logsumexp(safe, dim=-1)
        return torch.where(reachable, reduced, stacked.new_full((), -torch.inf))

    def forward(self, emissions: Tensor, termination_logits: Tensor) -> SegmentalAlignmentOutput:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [batch, states, frames]")
        batch, states, frames = emissions.shape
        if states != self.max_states:
            raise ValueError(f"expected {self.max_states} states, got {states}")
        if termination_logits.shape != (batch, states):
            raise ValueError("termination_logits must have shape [batch, states]")
        if states > frames:
            raise ValueError(
                f"number of process states ({states}) exceeds frames ({frames}); "
                "each active prefix state must be visited"
            )

        negative = emissions.new_full((batch,), -torch.inf)
        initial = [emissions[:, 0, 0]] + [negative for _ in range(states - 1)]
        alpha_steps = [torch.stack(initial, dim=-1)]
        for time in range(1, frames):
            previous = alpha_steps[-1]
            current = []
            for state in range(states):
                candidates = [previous[:, state] + self.stay[state]]
                if state > 0:
                    candidates.append(
                        previous[:, state - 1] + self.advance[state - 1]
                    )
                current.append(
                    emissions[:, state, time] + self._safe_logsumexp(candidates)
                )
            alpha_steps.append(torch.stack(current, dim=-1))
        alpha = torch.stack(alpha_steps, dim=1)

        log_z = alpha[:, -1, :]
        log_prior = termination_logits.log_softmax(dim=-1)
        joint_terminal = log_prior + log_z
        score = torch.logsumexp(joint_terminal, dim=-1)
        terminal_posterior = (joint_terminal - score.unsqueeze(-1)).exp()
        terminal_prior = log_prior.exp()

        posterior = emissions.new_zeros((batch, frames, states))
        for terminal in range(states):
            final = [negative for _ in range(states)]
            final[terminal] = emissions.new_zeros((batch,))
            beta_steps = [torch.stack(final, dim=-1)]
            for time in range(frames - 2, -1, -1):
                following = beta_steps[-1]
                current = []
                for state in range(terminal + 1):
                    candidates = [
                        self.stay[state]
                        + emissions[:, state, time + 1]
                        + following[:, state]
                    ]
                    if state < terminal:
                        candidates.append(
                            self.advance[state]
                            + emissions[:, state + 1, time + 1]
                            + following[:, state + 1]
                        )
                    current.append(
                        self._safe_logsumexp(candidates)
                    )
                current.extend(negative for _ in range(states - terminal - 1))
                beta_steps.append(torch.stack(current, dim=-1))
            beta = torch.stack(list(reversed(beta_steps)), dim=1)
            conditional = (
                alpha + beta - log_z[:, terminal].view(batch, 1, 1)
            ).exp()
            posterior = posterior + conditional * terminal_posterior[:, terminal].view(
                batch, 1, 1
            )

        lengths = torch.arange(1, states + 1, device=emissions.device, dtype=emissions.dtype)
        expected_length = (terminal_prior * lengths).sum(dim=-1)
        expected_duration = posterior.sum(dim=1)
        posterior_entropy = -(
            posterior.clamp_min(torch.finfo(posterior.dtype).tiny)
            * posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()
        ).sum(dim=-1).mean(dim=-1)
        return SegmentalAlignmentOutput(
            score=score,
            log_z_by_terminal=log_z,
            terminal_prior=terminal_prior,
            terminal_posterior=terminal_posterior,
            posterior=posterior,
            expected_length=expected_length,
            expected_duration=expected_duration,
            posterior_entropy=posterior_entropy,
        )

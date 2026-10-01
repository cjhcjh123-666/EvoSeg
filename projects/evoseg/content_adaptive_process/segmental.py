"""Content-adaptive PRE/process/POST semi-Markov forward-backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def stick_breaking_probabilities(halt_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Return prefix-consistent ``P(N|q)`` and per-slot halt probabilities.

    The last state receives all remaining probability, so the probabilities
    sum to one without a free softmax over unrelated terminal lengths.
    """

    if halt_logits.ndim != 2 or halt_logits.shape[1] < 1:
        raise ValueError("halt_logits must have shape [batch, max_states]")
    halt = halt_logits.sigmoid()
    batch, states = halt.shape
    survival = halt.new_ones(batch)
    probabilities = []
    for state in range(states - 1):
        probabilities.append(survival * halt[:, state])
        survival = survival * (1.0 - halt[:, state])
    probabilities.append(survival)
    effective_halt = torch.cat([halt[:, :-1], halt.new_ones(batch, 1)], dim=-1)
    return torch.stack(probabilities, dim=-1), effective_halt


@dataclass
class AdaptiveAlignmentOutput:
    score: Tensor
    log_z_by_length: Tensor
    length_prior: Tensor
    length_posterior: Tensor
    posterior_all: Tensor
    posterior_pre: Tensor
    posterior_process: Tensor
    posterior_post: Tensor
    process_occupancy: Tensor
    start_posterior: Tensor
    end_posterior: Tensor
    transition_confidence: Tensor
    expected_length: Tensor
    expected_duration: Tensor
    posterior_entropy: Tensor


class AdaptiveSegmentalForwardBackward(nn.Module):
    """Marginalize variable-length PRE→p1…pN→POST paths in log space."""

    def __init__(self, max_states: int = 6) -> None:
        super().__init__()
        if max_states < 1:
            raise ValueError("max_states must be positive")
        self.max_states = max_states

    @staticmethod
    def _lse(values: list[Tensor]) -> Tensor:
        stacked = torch.stack(values, dim=-1)
        reachable = torch.isfinite(stacked).any(dim=-1)
        floor = torch.finfo(stacked.dtype).min
        safe = stacked.masked_fill(~torch.isfinite(stacked), floor)
        value = torch.logsumexp(safe, dim=-1)
        return torch.where(reachable, value, stacked.new_full((), -torch.inf))

    @staticmethod
    def _binary_log_probs(logits: Tensor) -> Tensor:
        return torch.stack(
            [torch.nn.functional.logsigmoid(-logits), torch.nn.functional.logsigmoid(logits)],
            dim=-1,
        )

    def forward(
        self,
        emissions: Tensor,
        length_prior: Tensor,
        start_logits: Tensor,
        transition_logits: Tensor,
        end_logits: Tensor,
    ) -> AdaptiveAlignmentOutput:
        """Run content-adaptive forward-backward.

        Args:
            emissions: ``[B,M,T]`` process-state/frame scores.
            length_prior: ``[B,M]`` stick-breaking probabilities.
            start_logits: ``[B,T]`` PRE→p1 versus PRE stay logits.
            transition_logits: ``[B,M,T-1,2]`` stay/advance logits.
            end_logits: ``[B,M,T-1,2]`` pN stay/end logits.
        """

        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [batch, states, frames]")
        batch, states, frames = emissions.shape
        expected_shapes = {
            "length_prior": (batch, states),
            "start_logits": (batch, frames),
            "transition_logits": (batch, states, max(0, frames - 1), 2),
            "end_logits": (batch, states, max(0, frames - 1), 2),
        }
        values = {
            "length_prior": length_prior.shape,
            "start_logits": start_logits.shape,
            "transition_logits": transition_logits.shape,
            "end_logits": end_logits.shape,
        }
        for name, shape in expected_shapes.items():
            if values[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values[name]}")
        if states != self.max_states:
            raise ValueError(f"expected {self.max_states} process states, got {states}")
        if frames < 1:
            raise ValueError("at least one frame is required")

        start_lp = self._binary_log_probs(start_logits)
        transition_lp = transition_logits.log_softmax(dim=-1)
        end_lp = end_logits.log_softmax(dim=-1)
        negative = emissions.new_full((batch,), -torch.inf)
        alpha_by_length: list[Tensor] = []
        beta_by_length: list[Tensor] = []
        log_z_values: list[Tensor] = []

        # The shared full state axis is PRE, p1..pM, POST.
        full_states = states + 2
        post_index = states + 1
        for length in range(1, states + 1):
            if length > frames:
                alpha_by_length.append(
                    emissions.new_full((batch, frames, full_states), -torch.inf)
                )
                beta_by_length.append(
                    emissions.new_full((batch, frames, full_states), -torch.inf)
                )
                log_z_values.append(negative)
                continue

            initial = [negative for _ in range(full_states)]
            initial[0] = start_lp[:, 0, 0]
            initial[1] = start_lp[:, 0, 1] + emissions[:, 0, 0]
            alpha_steps = [torch.stack(initial, dim=-1)]
            for time in range(1, frames):
                previous = alpha_steps[-1]
                current = [negative for _ in range(full_states)]
                current[0] = previous[:, 0] + start_lp[:, time, 0]
                for process_state in range(length):
                    full_index = process_state + 1
                    if process_state == length - 1:
                        stay = end_lp[:, process_state, time - 1, 0]
                    else:
                        stay = transition_lp[:, process_state, time - 1, 0]
                    candidates = [previous[:, full_index] + stay]
                    if process_state == 0:
                        candidates.append(previous[:, 0] + start_lp[:, time, 1])
                    else:
                        candidates.append(
                            previous[:, full_index - 1]
                            + transition_lp[:, process_state - 1, time - 1, 1]
                        )
                    current[full_index] = emissions[:, process_state, time] + self._lse(candidates)
                current[post_index] = self._lse(
                    [
                        previous[:, post_index],
                        previous[:, length] + end_lp[:, length - 1, time - 1, 1],
                    ]
                )
                alpha_steps.append(torch.stack(current, dim=-1))
            alpha = torch.stack(alpha_steps, dim=1)
            log_z = self._lse([alpha[:, -1, length], alpha[:, -1, post_index]])

            final = [negative for _ in range(full_states)]
            final[length] = emissions.new_zeros(batch)
            final[post_index] = emissions.new_zeros(batch)
            beta_steps = [torch.stack(final, dim=-1)]
            for time in range(frames - 2, -1, -1):
                following = beta_steps[-1]
                current = [negative for _ in range(full_states)]
                current[0] = self._lse(
                    [
                        start_lp[:, time + 1, 0] + following[:, 0],
                        start_lp[:, time + 1, 1]
                        + emissions[:, 0, time + 1]
                        + following[:, 1],
                    ]
                )
                for process_state in range(length):
                    full_index = process_state + 1
                    if process_state == length - 1:
                        current[full_index] = self._lse(
                            [
                                end_lp[:, process_state, time, 0]
                                + emissions[:, process_state, time + 1]
                                + following[:, full_index],
                                end_lp[:, process_state, time, 1]
                                + following[:, post_index],
                            ]
                        )
                    else:
                        current[full_index] = self._lse(
                            [
                                transition_lp[:, process_state, time, 0]
                                + emissions[:, process_state, time + 1]
                                + following[:, full_index],
                                transition_lp[:, process_state, time, 1]
                                + emissions[:, process_state + 1, time + 1]
                                + following[:, full_index + 1],
                            ]
                        )
                current[post_index] = following[:, post_index]
                beta_steps.append(torch.stack(current, dim=-1))
            beta = torch.stack(list(reversed(beta_steps)), dim=1)
            alpha_by_length.append(alpha)
            beta_by_length.append(beta)
            log_z_values.append(log_z)

        log_z_by_length = torch.stack(log_z_values, dim=-1)
        log_length = length_prior.clamp_min(torch.finfo(length_prior.dtype).tiny).log()
        joint = log_length + log_z_by_length
        score = torch.logsumexp(joint, dim=-1)
        length_posterior = (joint - score.unsqueeze(-1)).exp()
        posterior_all = emissions.new_zeros(batch, frames, full_states)
        start_posterior = emissions.new_zeros(batch, frames)
        end_posterior = emissions.new_zeros(batch, frames)
        advance_posterior = emissions.new_zeros(batch, frames)
        for length in range(1, states + 1):
            if length > frames:
                continue
            alpha = alpha_by_length[length - 1]
            beta = beta_by_length[length - 1]
            conditional = (alpha + beta - log_z_by_length[:, length - 1].view(batch, 1, 1)).exp()
            weight = length_posterior[:, length - 1].view(batch, 1, 1)
            posterior_all = posterior_all + conditional * weight

            start0 = (
                start_lp[:, 0, 1]
                + emissions[:, 0, 0]
                + beta[:, 0, 1]
                - log_z_by_length[:, length - 1]
            ).exp()
            start_posterior[:, 0] += start0 * length_posterior[:, length - 1]
            # A process still occupying p_N on the final frame ends at the
            # observed video boundary rather than having an unobserved POST.
            end_posterior[:, -1] += (
                conditional[:, -1, length] * length_posterior[:, length - 1]
            )
            for time in range(1, frames):
                start_value = (
                    alpha[:, time - 1, 0]
                    + start_lp[:, time, 1]
                    + emissions[:, 0, time]
                    + beta[:, time, 1]
                    - log_z_by_length[:, length - 1]
                ).exp()
                start_posterior[:, time] += start_value * length_posterior[:, length - 1]
                end_value = (
                    alpha[:, time - 1, length]
                    + end_lp[:, length - 1, time - 1, 1]
                    + beta[:, time, post_index]
                    - log_z_by_length[:, length - 1]
                ).exp()
                end_posterior[:, time] += end_value * length_posterior[:, length - 1]
                for process_state in range(length - 1):
                    advance_value = (
                        alpha[:, time - 1, process_state + 1]
                        + transition_lp[:, process_state, time - 1, 1]
                        + emissions[:, process_state + 1, time]
                        + beta[:, time, process_state + 2]
                        - log_z_by_length[:, length - 1]
                    ).exp()
                    advance_posterior[:, time] += (
                        advance_value * length_posterior[:, length - 1]
                    )

        posterior_process = posterior_all[:, :, 1 : states + 1]
        occupancy = posterior_process.sum(dim=-1)
        transition_confidence = (start_posterior + end_posterior + advance_posterior).clamp_max(1.0)
        lengths = torch.arange(1, states + 1, device=emissions.device, dtype=emissions.dtype)
        expected_length = (length_prior * lengths).sum(dim=-1)
        values = posterior_all.clamp_min(torch.finfo(posterior_all.dtype).tiny)
        entropy = -(values * values.log()).sum(dim=-1).mean(dim=-1)
        return AdaptiveAlignmentOutput(
            score=score,
            log_z_by_length=log_z_by_length,
            length_prior=length_prior,
            length_posterior=length_posterior,
            posterior_all=posterior_all,
            posterior_pre=posterior_all[:, :, 0],
            posterior_process=posterior_process,
            posterior_post=posterior_all[:, :, post_index],
            process_occupancy=occupancy,
            start_posterior=start_posterior,
            end_posterior=end_posterior,
            transition_confidence=transition_confidence,
            expected_length=expected_length,
            expected_duration=posterior_process.sum(dim=1),
            posterior_entropy=entropy,
        )

"""Differentiable, validity-gated monotonic process/frame alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MonotonicAlignmentOutput:
    score: Tensor
    posterior: Tensor
    aligned_mass: Tensor


class MonotonicAlignment(nn.Module):
    """Log-sum-exp DP over optional process slots and strictly ordered frames.

    A process slot can either be skipped or align to one frame. Aligned slots must
    use strictly increasing frame indices. The returned posterior is the exact
    marginal probability that slot ``m`` aligns to frame ``t`` under this model.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, compatibility: Tensor, validity: Tensor) -> MonotonicAlignmentOutput:
        if compatibility.ndim != 3:
            raise ValueError("compatibility must have shape [batch, slots, frames]")
        if validity.shape != compatibility.shape[:2]:
            raise ValueError("validity must have shape [batch, slots]")

        batch, slots, frames = compatibility.shape
        if frames < 1:
            raise ValueError("at least one frame is required")

        dtype = compatibility.dtype
        device = compatibility.device
        neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)
        gate = validity.clamp(self.eps, 1.0 - self.eps)
        log_align = gate.log()
        log_skip = torch.log1p(-gate)
        # Exact gates are useful for deterministic ablations and make an empty
        # slot a true no-op, independent of the magnitude of its compatibility.
        log_align = torch.where(validity <= self.eps, neg_inf, log_align)
        log_skip = torch.where(validity >= 1.0 - self.eps, neg_inf, log_skip)

        # State zero means no event has been aligned; state t+1 means frame t was
        # the most recently aligned frame.
        forward = [torch.full((batch, frames + 1), neg_inf, dtype=dtype, device=device)]
        forward[0][:, 0] = 0.0
        for m in range(slots):
            prev = forward[-1]
            cur = prev + log_skip[:, m : m + 1]
            aligned_states = []
            for t in range(frames):
                valid_prev = torch.logsumexp(prev[:, : t + 1], dim=-1)
                aligned_states.append(valid_prev + log_align[:, m] + compatibility[:, m, t])
            aligned = torch.stack(aligned_states, dim=-1)
            cur = torch.cat(
                [cur[:, :1], torch.logaddexp(cur[:, 1:], aligned)], dim=-1
            )
            forward.append(cur)

        backward = [None] * (slots + 1)
        backward[slots] = torch.zeros((batch, frames + 1), dtype=dtype, device=device)
        for m in range(slots - 1, -1, -1):
            nxt = backward[m + 1]
            states = []
            for state in range(frames + 1):
                terms = [log_skip[:, m] + nxt[:, state]]
                first_frame = state
                if first_frame < frames:
                    terms.append(
                        torch.logsumexp(
                            log_align[:, m : m + 1]
                            + compatibility[:, m, first_frame:]
                            + nxt[:, first_frame + 1 :],
                            dim=-1,
                        )
                    )
                states.append(torch.logsumexp(torch.stack(terms, dim=-1), dim=-1))
            backward[m] = torch.stack(states, dim=-1)

        score = torch.logsumexp(forward[-1], dim=-1)
        posterior_slots = []
        for m in range(slots):
            frame_probs = []
            for t in range(frames):
                prefix = torch.logsumexp(forward[m][:, : t + 1], dim=-1)
                log_prob = (
                    prefix
                    + log_align[:, m]
                    + compatibility[:, m, t]
                    + backward[m + 1][:, t + 1]
                    - score
                )
                frame_probs.append(log_prob.exp())
            posterior_slots.append(torch.stack(frame_probs, dim=-1))
        posterior = torch.stack(posterior_slots, dim=1)
        return MonotonicAlignmentOutput(
            score=score,
            posterior=posterior,
            aligned_mass=posterior.sum(dim=-1),
        )

"""Identity/state factorization for the public VIRST SegPrompter.

The wrapper is deliberately non-invasive: it calls the released VIRST
``SegPrompter`` first, then factorizes its already trained frame prompts.  At
initialization the FTG variant is exactly the released model, which makes VIRST
a controlled strong foundation instead of asking a randomly initialized prompt
interface to rediscover the public checkpoint's language-to-mask geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


def pad_video_frames_to_multiple(
    frames: Tensor,
    multiple: int = 4,
) -> Tensor:
    """Repeat the final VLM frame for VideoChat's fixed local-frame groups."""
    if frames.ndim < 1 or len(frames) == 0:
        raise ValueError("video frame tensor must be non-empty")
    if multiple <= 0:
        raise ValueError("frame multiple must be positive")
    missing = (-len(frames)) % multiple
    if missing == 0:
        return frames
    repeat_shape = (missing,) + (1,) * (frames.ndim - 1)
    padding = frames[-1:].repeat(repeat_shape)
    return torch.cat([frames, padding], dim=0)


VirstGroundingVariant = Literal[
    "public_virst",
    "identity_only",
    "state_only",
    "factorized_no_gate",
    "ftg",
    "unconditioned_ftg",
    "scalar_ftg",
]


@dataclass
class VirstFactorizationDiagnostics:
    identity: Tensor
    state: Tensor
    gate: Tensor
    prompt: Tensor


class IdentityStateComposer(nn.Module):
    """Recompose frame prompts while preserving their temporal identity mean.

    Given public frame prompts ``x_t``, the persistent identity and dynamic state
    are defined without extra supervision:

    ``z_id = mean_t(x_t)`` and ``z_state_t = x_t - z_id``.

    The learned FTG gate starts at one, so ``z_id + z_state_t == x_t`` exactly.
    After gating, the dynamic branch is centered again.  Consequently the mean
    of every FTG prompt sequence remains exactly ``z_id`` throughout training;
    the trainable path can change temporal state but cannot rewrite identity.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_gate_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 < max_gate_delta <= 1:
            raise ValueError("max_gate_delta must be in (0, 1]")
        self.hidden_dim = hidden_dim
        self.max_gate_delta = max_gate_delta
        self.identity_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # tanh(0) = 0, hence the multiplicative gate is exactly one and the
        # public VIRST prompt is preserved before any optimization step.
        nn.init.zeros_(self.vector_gate[-1].weight)
        nn.init.zeros_(self.vector_gate[-1].bias)
        nn.init.zeros_(self.scalar_gate[-1].weight)
        nn.init.zeros_(self.scalar_gate[-1].bias)

    @staticmethod
    def factorize(prompts: Tensor) -> tuple[Tensor, Tensor]:
        if prompts.ndim != 4:
            raise ValueError("VIRST prompts must have shape [B, N, T, C]")
        identity = prompts.mean(dim=2, keepdim=True)
        state = prompts - identity
        return identity, state

    def forward(
        self,
        prompts: Tensor,
        variant: VirstGroundingVariant = "ftg",
    ) -> tuple[Tensor, VirstFactorizationDiagnostics]:
        identity, state = self.factorize(prompts)
        if prompts.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected prompt dim {self.hidden_dim}, got {prompts.shape[-1]}"
            )
        expanded_identity = identity.expand_as(prompts)
        inputs = torch.cat(
            [self.identity_norm(expanded_identity), self.state_norm(state)],
            dim=-1,
        )

        if variant == "public_virst":
            gate = torch.ones_like(state)
            output = prompts
        elif variant == "identity_only":
            gate = torch.zeros_like(state)
            output = expanded_identity
        elif variant == "state_only":
            gate = torch.ones_like(state)
            output = state
        elif variant == "factorized_no_gate":
            gate = torch.ones_like(state)
            output = prompts
        elif variant in {"ftg", "unconditioned_ftg", "scalar_ftg"}:
            if variant == "unconditioned_ftg":
                inputs = torch.cat(
                    [torch.zeros_like(expanded_identity), self.state_norm(state)],
                    dim=-1,
                )
            gate_network = (
                self.scalar_gate if variant == "scalar_ftg" else self.vector_gate
            )
            gate_delta = self.max_gate_delta * torch.tanh(gate_network(inputs))
            gate = 1.0 + gate_delta
            # Express the learned operation as a correction to the released
            # prompt. When gate_delta is zero this adds literal floating-point
            # zeros, rather than subtracting and re-adding the temporal mean.
            # That gives a bit-exact public-model initialization.
            correction = gate_delta * state
            # A varying gate could otherwise leak a time-constant component
            # into the state path. Re-centering keeps identity structurally fixed.
            correction = correction - correction.mean(dim=2, keepdim=True)
            output = prompts + correction
        else:
            raise ValueError(f"unknown VIRST grounding variant: {variant}")

        diagnostics = VirstFactorizationDiagnostics(
            identity=identity,
            state=state,
            gate=gate,
            prompt=output,
        )
        return output, diagnostics


class FactorizedVirstSegPrompter(nn.Module):
    """Wrap the released VIRST prompter and preserve its public API."""

    def __init__(
        self,
        official: nn.Module,
        variant: VirstGroundingVariant = "ftg",
        max_gate_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.official = official
        self.variant = variant
        for attribute in ("token_dim", "max_length", "grid_hw", "nhead"):
            if hasattr(official, attribute):
                setattr(self, attribute, getattr(official, attribute))
        self.composer = IdentityStateComposer(
            hidden_dim=official.token_dim,
            max_gate_delta=max_gate_delta,
        )
        self.last_diagnostics: VirstFactorizationDiagnostics | None = None

    def forward(self, *args, **kwargs):
        result = self.official(*args, **kwargs)
        if isinstance(result, tuple):
            prompts, frame_scores = result
        else:
            prompts, frame_scores = result, None
        prompts, self.last_diagnostics = self.composer(
            prompts,
            variant=self.variant,
        )
        if frame_scores is None:
            return prompts
        return prompts, frame_scores


def find_virst_core(model: nn.Module) -> nn.Module:
    """Resolve the one VIRST core through PEFT/DeepSpeed wrapper layers."""
    matches = []
    for module in model.modules():
        nested = getattr(module, "model", None)
        if (
            hasattr(module, "seg_token_idx")
            and nested is not None
            and hasattr(nested, "seg_prompter")
        ):
            matches.append(module)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one VIRST core, found {len(matches)}")
    return matches[0]


def install_virst_ftg(
    model: nn.Module,
    variant: VirstGroundingVariant = "ftg",
    max_gate_delta: float = 0.5,
) -> FactorizedVirstSegPrompter:
    """Install FTG after the public VIRST checkpoint has been loaded."""
    core = find_virst_core(model)
    if isinstance(core.model.seg_prompter, FactorizedVirstSegPrompter):
        raise RuntimeError("VIRST FTG is already installed")
    wrapper = FactorizedVirstSegPrompter(
        core.model.seg_prompter,
        variant=variant,
        max_gate_delta=max_gate_delta,
    )
    core.model.seg_prompter = wrapper
    return wrapper


def ftg_state_dict(wrapper: FactorizedVirstSegPrompter) -> dict[str, Tensor]:
    """Return only the small FTG parameters, excluding public VIRST weights."""
    return wrapper.composer.state_dict()

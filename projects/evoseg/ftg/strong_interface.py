"""Identity/state prompt roles for the pretrained Qwen3-VL/SAM3 foundation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


StrongGroundingVariant = Literal[
    "identity_memory",
    "state_only",
    "frame_prompt",
    "vector_sum",
    "ftg",
    "anchored_ftg",
    "unconditioned_residual",
    "bounded_ftg",
    "normalized_ftg",
    "centered_ftg",
    "gated_centered_ftg",
    "vector_gated_centered_ftg",
]


class FactorizedPromptTokens(nn.Module):
    """Compose native SAM sparse tokens while keeping identity structurally fixed.

    ``identity`` is the pretrained Sa2VA ``[SEG]`` projection. ``frame_features``
    are native SAM spatial features aligned one-to-one with the repeated identity
    rows. A single identity query cross-attends to each frame, so the state is
    target-aware rather than a global average. FTG returns identity and state as
    two sparse prompt tokens; the
    ``vector_sum`` control deliberately collapses them back into one token.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_residual_ratio: float = 0.02,
    ) -> None:
        super().__init__()
        if not 0 < max_residual_ratio <= 1:
            raise ValueError("max_residual_ratio must be in (0, 1]")
        self.max_residual_ratio = max_residual_ratio
        self.identity_norm = nn.LayerNorm(hidden_dim)
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.state_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8 if hidden_dim % 8 == 0 else 1,
            batch_first=True,
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # The strong-foundation path must begin as the published model, not as
        # a randomly perturbed prompt.  This separate residual is exactly zero
        # at initialization and is the only new content used by anchored FTG.
        self.anchored_state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.anchored_state_mlp[-1].weight)
        nn.init.zeros_(self.anchored_state_mlp[-1].bias)
        self.state_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.state_vector_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.state_vector_gate[-1].weight)
        nn.init.zeros_(self.state_vector_gate[-1].bias)

    def state_observation(
        self,
        identity: torch.Tensor,
        frame_features: torch.Tensor,
        *,
        anchored: bool = False,
        centered: bool = False,
        center_output: bool = False,
        vector_gate: bool = False,
        gate_after_norm: bool = False,
        group_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if identity.ndim != 2 or frame_features.ndim != 3:
            raise ValueError("identity must be [N, C] and frame_features [N, S, C]")
        if identity.shape[0] != frame_features.shape[0] or \
                identity.shape[1] != frame_features.shape[2]:
            raise ValueError("identity and frame_features batch/channel shapes differ")
        query = self.identity_norm(identity)[:, None]
        frame_features = self.frame_norm(frame_features)
        observation, _ = self.state_attention(
            query, frame_features, frame_features, need_weights=False
        )
        observation = observation[:, 0]
        if centered:
            if group_shape is None or group_shape[0] * group_shape[1] != len(identity):
                raise ValueError(
                    "centered state requires group_shape=(num_frames, num_objects)"
                )
            num_frames, num_objects = group_shape
            observation = observation.reshape(num_frames, num_objects, -1)
            observation = observation - observation.mean(dim=0, keepdim=True)
            observation = observation.flatten(0, 1)
        inputs = torch.cat(
            [self.identity_norm(identity), observation], dim=-1
        )
        gate_module = self.state_vector_gate if vector_gate else self.state_gate
        gate = gate_module(inputs).sigmoid()
        state_mlp = self.anchored_state_mlp if anchored else self.state_mlp
        state = state_mlp(inputs)
        if not gate_after_norm:
            state = gate * state
        if center_output:
            num_frames, num_objects = group_shape
            state = state.reshape(num_frames, num_objects, -1)
            state = state - state.mean(dim=0, keepdim=True)
            state = state.flatten(0, 1)
        return state, gate

    def forward(
        self,
        identity: torch.Tensor,
        frame_features: torch.Tensor,
        variant: StrongGroundingVariant = "ftg",
        group_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchored = variant in {
            "anchored_ftg", "unconditioned_residual", "bounded_ftg",
            "normalized_ftg",
            "centered_ftg",
            "gated_centered_ftg",
            "vector_gated_centered_ftg",
        }
        state_identity = (
            torch.zeros_like(identity)
            if variant == "unconditioned_residual"
            else identity
        )
        state, gate = self.state_observation(
            state_identity,
            frame_features,
            anchored=anchored,
            centered=variant in {
                "centered_ftg", "gated_centered_ftg",
                "vector_gated_centered_ftg",
            },
            center_output=variant in {
                "gated_centered_ftg", "vector_gated_centered_ftg",
            },
            vector_gate=variant == "vector_gated_centered_ftg",
            gate_after_norm=variant == "vector_gated_centered_ftg",
            group_shape=group_shape,
        )
        if variant == "identity_memory":
            tokens = identity[:, None]
        elif variant == "state_only":
            state, gate = self.state_observation(
                torch.zeros_like(identity), frame_features
            )
            tokens = state[:, None]
        elif variant == "frame_prompt":
            tokens = state[:, None]
        elif variant == "vector_sum":
            tokens = (identity + state)[:, None]
        elif variant == "ftg":
            tokens = torch.stack([identity, state], dim=1)
        elif variant == "anchored_ftg":
            # One sparse prompt retains the pretrained identity geometry.  The
            # target-aware dynamic state can move it only after learning; at
            # initialization this is bit-for-bit the identity-memory prompt.
            tokens = (identity + state)[:, None]
        elif variant == "unconditioned_residual":
            # Parameter-matched control: preserve the same foundation prompt
            # and residual branch, but remove identity from state extraction.
            tokens = (identity + state)[:, None]
        elif variant == "bounded_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            max_norm = self.max_residual_ratio * identity_norm
            residual_scale = torch.clamp(
                max_norm / state_norm.clamp_min(1e-6), max=1.0
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        elif variant == "normalized_ftg":
            # Treat the learned state as a direction and allocate it an exact
            # fraction of the identity norm.  Unlike bounded_ftg, this exposes
            # useful but naturally small state vectors instead of only clipping
            # vectors that happen to be too large.
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        elif variant == "centered_ftg":
            # Remove the per-object temporal mean before producing state, so
            # this branch cannot redundantly encode persistent appearance.
            # The remaining dynamic direction receives an explicit prompt
            # budget relative to the stable identity base.
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        elif variant == "gated_centered_ftg":
            # Center both the frame observation and the produced state over
            # time. Unlike exact-normalized controls, retain the learned gate
            # *after* direction normalization so it controls residual amplitude
            # from zero to the configured identity-relative budget.
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + gate * residual_scale * state)[:, None]
        elif variant == "vector_gated_centered_ftg":
            # Channel-wise post-normalization gating implements the intended
            # element-wise composition while keeping a hard 2% norm ceiling:
            # sigmoid gates lie in [0, 1], so they can only shrink the
            # normalized dynamic direction.
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + gate * residual_scale * state)[:, None]
        else:
            raise ValueError(f"unknown strong grounding variant: {variant}")
        return tokens, gate

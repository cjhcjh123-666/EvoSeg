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

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
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

    def state_observation(
        self,
        identity: torch.Tensor,
        frame_features: torch.Tensor,
        *,
        anchored: bool = False,
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
        inputs = torch.cat(
            [self.identity_norm(identity), observation], dim=-1
        )
        gate = self.state_gate(inputs).sigmoid()
        state_mlp = self.anchored_state_mlp if anchored else self.state_mlp
        state = gate * state_mlp(inputs)
        return state, gate

    def forward(
        self,
        identity: torch.Tensor,
        frame_features: torch.Tensor,
        variant: StrongGroundingVariant = "ftg",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchored = variant in {"anchored_ftg", "unconditioned_residual"}
        state_identity = (
            torch.zeros_like(identity)
            if variant == "unconditioned_residual"
            else identity
        )
        state, gate = self.state_observation(
            state_identity, frame_features, anchored=anchored
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
        else:
            raise ValueError(f"unknown strong grounding variant: {variant}")
        return tokens, gate

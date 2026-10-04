"""Standalone FTG prompt composer copied into exported Hugging Face models."""

import torch
from torch import nn


class FactorizedPromptTokens(nn.Module):
    def __init__(self, hidden_dim=256, max_residual_ratio=0.02):
        super().__init__()
        if not 0 < max_residual_ratio <= 1:
            raise ValueError("max_residual_ratio must be in (0, 1]")
        self.max_residual_ratio = max_residual_ratio
        self.identity_norm = nn.LayerNorm(hidden_dim)
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.state_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8 if hidden_dim % 8 == 0 else 1,
            batch_first=True,
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
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

    def state_observation(self, identity, frame_features, *, anchored=False):
        query = self.identity_norm(identity)[:, None]
        frame_features = self.frame_norm(frame_features)
        observation, _ = self.state_attention(
            query, frame_features, frame_features, need_weights=False
        )
        inputs = torch.cat([self.identity_norm(identity), observation[:, 0]], -1)
        gate = self.state_gate(inputs).sigmoid()
        state_mlp = self.anchored_state_mlp if anchored else self.state_mlp
        return gate * state_mlp(inputs), gate

    def forward(self, identity, frame_features, variant="ftg"):
        state_identity = (
            torch.zeros_like(identity)
            if variant == "unconditioned_residual"
            else identity
        )
        state, gate = self.state_observation(
            state_identity,
            frame_features,
            anchored=variant in {
                "anchored_ftg", "unconditioned_residual", "bounded_ftg"
            },
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
            tokens = torch.stack([identity, state], 1)
        elif variant == "anchored_ftg":
            tokens = (identity + state)[:, None]
        elif variant == "unconditioned_residual":
            tokens = (identity + state)[:, None]
        elif variant == "bounded_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            max_norm = self.max_residual_ratio * identity_norm
            residual_scale = torch.clamp(
                max_norm / state_norm.clamp_min(1e-6), max=1.0
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        else:
            raise ValueError(f"unknown grounding variant: {variant}")
        return tokens, gate

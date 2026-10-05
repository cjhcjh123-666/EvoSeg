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
        self.state_vector_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.state_vector_gate[-1].weight)
        nn.init.zeros_(self.state_vector_gate[-1].bias)

    def state_observation(
        self, identity, frame_features, *, anchored=False, centered=False,
        center_output=False, vector_gate=False, gate_after_norm=False,
        group_shape=None,
    ):
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
        inputs = torch.cat([self.identity_norm(identity), observation], -1)
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
        self, identity, frame_features, variant="ftg", group_shape=None
    ):
        state_identity = (
            torch.zeros_like(identity)
            if variant == "unconditioned_residual"
            else identity
        )
        state, gate = self.state_observation(
            state_identity,
            frame_features,
            anchored=variant in {
                "anchored_ftg", "unconditioned_residual", "bounded_ftg",
                "normalized_ftg",
                "centered_ftg",
                "gated_centered_ftg",
                "vector_gated_centered_ftg",
            },
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
        elif variant == "normalized_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        elif variant == "centered_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + residual_scale * state)[:, None]
        elif variant == "gated_centered_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + gate * residual_scale * state)[:, None]
        elif variant == "vector_gated_centered_ftg":
            identity_norm = identity.float().norm(dim=-1, keepdim=True)
            state_norm = state.float().norm(dim=-1, keepdim=True)
            residual_scale = (
                self.max_residual_ratio * identity_norm
                / state_norm.clamp_min(1e-6)
            ).to(state.dtype)
            tokens = (identity + gate * residual_scale * state)[:, None]
        else:
            raise ValueError(f"unknown grounding variant: {variant}")
        return tokens, gate

"""Identity/state grounding interfaces shared by every controlled FTG variant."""

from __future__ import annotations

import torch
from torch import nn


FTG_VARIANTS = (
    "global_prompt",
    "frame_prompt",
    "identity_only",
    "state_only",
    "id_state_no_gate",
    "ftg",
)


class ResidualConditioner(nn.Module):
    """Produce one residual state from two equally sized conditioning states."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, primary: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if primary.shape != condition.shape:
            raise ValueError(
                f"primary and condition must match, got {primary.shape} and {condition.shape}"
            )
        return primary + self.net(torch.cat([primary, condition], dim=-1))


class PromptProjection(nn.Module):
    def __init__(self, hidden_dim: int, prompt_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, prompt_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class FactorizedTemporalGrounding(nn.Module):
    """Compose a persistent identity base with frame-dependent state residuals.

    The six modes instantiate the controlled comparison under one implementation:

    - ``global_prompt``: one monolithic video/query prompt for every frame;
    - ``frame_prompt``: independent frame/query prompts;
    - ``identity_only``: one explicit persistent identity prompt;
    - ``state_only``: frame states without persistent identity conditioning;
    - ``id_state_no_gate``: identity plus an ungated state residual;
    - ``ftg``: identity plus a learned, element-wise gated state residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        prompt_dim: int = 256,
        bottleneck_dim: int = 512,
        variant: str = "ftg",
    ) -> None:
        super().__init__()
        if variant not in FTG_VARIANTS:
            raise ValueError(f"unknown FTG variant {variant!r}; expected {FTG_VARIANTS}")
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim

        # Monolithic controls.
        self.global_encoder = ResidualConditioner(hidden_dim, bottleneck_dim)
        self.frame_encoder = ResidualConditioner(hidden_dim, bottleneck_dim)
        self.monolithic_projection = PromptProjection(
            hidden_dim, prompt_dim, bottleneck_dim
        )

        # Factorized path. Identity sees full-video context once. Dynamic state is
        # then computed per frame, conditioned on that fixed identity.
        self.identity_encoder = ResidualConditioner(hidden_dim, bottleneck_dim)
        self.state_encoder = ResidualConditioner(hidden_dim, bottleneck_dim)
        self.state_only_encoder = ResidualConditioner(hidden_dim, bottleneck_dim)
        self.identity_projection = PromptProjection(
            hidden_dim, prompt_dim, bottleneck_dim
        )
        self.state_projection = PromptProjection(
            hidden_dim, prompt_dim, bottleneck_dim
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, prompt_dim),
        )
        # FTG is explicitly an identity base plus a learned residual. Starting
        # that residual at zero makes the initial factorized prompt exactly the
        # identity-only prompt, so a random dynamic branch cannot cause identity
        # drift before it receives supervision.
        nn.init.zeros_(self.state_projection.net[-1].weight)
        nn.init.zeros_(self.state_projection.net[-1].bias)

        # State Only is an independent control, not a residual, and therefore
        # keeps a normally initialized projection of its own. Registering it
        # after the gate preserves initialization of every pre-existing branch.
        self.state_only_projection = PromptProjection(
            hidden_dim, prompt_dim, bottleneck_dim
        )

    @staticmethod
    def _expand(state: torch.Tensor, length: int) -> torch.Tensor:
        if state.ndim != 1:
            raise ValueError("global/query state must be one-dimensional")
        return state.unsqueeze(0).expand(length, -1)

    def forward(
        self,
        frame_states: torch.Tensor,
        query_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        if frame_states.ndim != 2:
            raise ValueError("frame_states must be [T,H]")
        if frame_states.shape[0] == 0:
            raise ValueError("at least one frame state is required")
        if query_state.ndim == 2 and query_state.shape[0] == 1:
            query_state = query_state[0]
        if query_state.shape != frame_states.shape[1:]:
            raise ValueError("query_state must be [H] and match frame hidden size")

        length = frame_states.shape[0]
        query_frames = self._expand(query_state, length)
        video_state = frame_states.mean(dim=0)

        identity = self.identity_encoder(query_state, video_state)
        identity_frames = self._expand(identity, length)
        dynamic_state = self.state_encoder(frame_states, identity_frames)
        independent_state = self.state_only_encoder(frame_states, query_frames)

        identity_prompt = self.identity_projection(identity)
        identity_prompts = self._expand(identity_prompt, length)
        state_prompts = self.state_projection(dynamic_state)
        independent_state_prompts = self.state_only_projection(independent_state)
        gate = torch.sigmoid(
            self.gate(torch.cat([identity_frames, dynamic_state], dim=-1))
        )

        if self.variant == "global_prompt":
            global_state = self.global_encoder(video_state, query_state)
            prompts = self._expand(self.monolithic_projection(global_state), length)
        elif self.variant == "frame_prompt":
            prompts = self.monolithic_projection(
                self.frame_encoder(frame_states, query_frames)
            )
        elif self.variant == "identity_only":
            prompts = identity_prompts
        elif self.variant == "state_only":
            prompts = independent_state_prompts
        elif self.variant == "id_state_no_gate":
            prompts = identity_prompts + state_prompts
        else:
            prompts = identity_prompts + gate * state_prompts

        diagnostics: dict[str, torch.Tensor | str] = {
            "grounding_variant": self.variant,
            "identity_token": identity,
            "state_tokens": (
                independent_state if self.variant == "state_only" else dynamic_state
            ),
            "state_gate": gate,
            "identity_prompts": identity_prompts,
            "state_prompts": state_prompts,
            "independent_state_prompts": independent_state_prompts,
            "identity_norm": identity.float().norm(),
            "state_norm": dynamic_state.float().norm(dim=-1).mean(),
            "gate_mean": gate.float().mean(),
            "gate_std": gate.float().std(),
            "prompt_cross_frame_std": prompts.float().std(dim=0).mean(),
        }
        return prompts, diagnostics

    def zero_active_output_projection(self) -> None:
        """Start a native-anchor residual at exactly zero for any variant."""
        projection = {
            "global_prompt": self.monolithic_projection,
            "frame_prompt": self.monolithic_projection,
            "identity_only": self.identity_projection,
            "state_only": self.state_only_projection,
            "id_state_no_gate": self.state_projection,
            "ftg": self.state_projection,
        }[self.variant]
        nn.init.zeros_(projection.net[-1].weight)
        nn.init.zeros_(projection.net[-1].bias)

    def active_parameter_count(self) -> int:
        """Count parameters receiving gradients for this controlled variant."""
        prefixes = self.active_parameter_prefixes()
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
        )

    def active_parameter_prefixes(self) -> tuple[str, ...]:
        return {
            "global_prompt": ("global_encoder", "monolithic_projection"),
            "frame_prompt": ("frame_encoder", "monolithic_projection"),
            "identity_only": ("identity_encoder", "identity_projection"),
            "state_only": ("state_only_encoder", "state_only_projection"),
            "id_state_no_gate": (
                "identity_encoder", "state_encoder",
                "identity_projection", "state_projection",
            ),
            "ftg": (
                "identity_encoder", "state_encoder",
                "identity_projection", "state_projection", "gate",
            ),
        }[self.variant]

    def activate_variant_parameters(self) -> list[nn.Parameter]:
        """Freeze inactive control branches and return only the active parameters."""
        prefixes = self.active_parameter_prefixes()
        active = []
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
            if parameter.requires_grad:
                active.append(parameter)
        return active

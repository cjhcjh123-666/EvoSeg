"""Untrained grounding-interface hypothesis, not a deployed or validated method."""
import torch
from torch import nn


def factorial_interaction(full, visual_only, query_only, neither):
    """Cancels additive unimodal components; does not prove causal identification.

    Neutral views must preserve token geometry and use no GT mask/target input.
    Out-of-distribution neutral inputs remain an empirical confound to test.
    """
    if not (full.shape == visual_only.shape == query_only.shape == neither.shape):
        raise ValueError('four grounding representations must have identical shapes')
    return full - visual_only - query_only + neither


class InteractionGroundingAdapter(nn.Module):
    def __init__(self, language_dim, prompt_dim=256, hidden_dim=256):
        super().__init__()
        self.features = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, hidden_dim), nn.GELU())
        self.prompt_residual = nn.Linear(hidden_dim, prompt_dim)
        self.empty_head = nn.Linear(hidden_dim, 1)
        # Preserve released prompt exactly before any method training.
        nn.init.zeros_(self.prompt_residual.weight)
        nn.init.zeros_(self.prompt_residual.bias)
        nn.init.zeros_(self.empty_head.weight)
        nn.init.constant_(self.empty_head.bias, -4.)

    def forward(self, base_prompt, full, visual_only, query_only, neither):
        difference = factorial_interaction(full, visual_only, query_only, neither)
        if base_prompt.shape[:-1] != difference.shape[:-1]:
            raise ValueError('base prompt and interaction batch dimensions differ')
        features = self.features(difference)
        return {'grounding_prompt': base_prompt + self.prompt_residual(features),
                'empty_logit': self.empty_head(features).squeeze(-1), 'interaction': difference}

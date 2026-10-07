"""Experimental learned spatial evidence; candidate quality never rejects masks.

This is a public-data pilot, not a validated novelty/SOTA claim. The frozen
Qwen query state is aligned with frozen SAM image features under human mask
supervision. Whole-image existence, candidate precision and prompt correction
have different outputs/labels. No raw similarity is forced into point prompts.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class EvidenceRoutingAdapter(nn.Module):
    def __init__(self, language_dim=4096, image_dim=256, hidden_dim=64, prompt_dim=256, native_prior_init=.1):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, hidden_dim), nn.GELU())
        self.image = nn.Conv2d(image_dim, hidden_dim, 1)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.map_bias = nn.Parameter(torch.zeros(()))
        # A binary native decision is NOT a calibrated high-confidence logit.
        # Learn its positive weight; small initialization preserves the initial
        # decision without preventing subsequent visual evidence from revising it.
        self.native_prior_log_weight = nn.Parameter(torch.tensor(math.log(math.expm1(native_prior_init))))
        self.global_fusion = nn.Sequential(nn.Linear(hidden_dim * 3 + 2, hidden_dim), nn.GELU())
        self.existence = nn.Linear(hidden_dim, 1)
        self.fallback = nn.Linear(hidden_dim, prompt_dim)
        self.prompt = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, hidden_dim), nn.GELU())
        self.candidate_fusion = nn.Sequential(nn.Linear(hidden_dim * 4 + 2, hidden_dim), nn.GELU())
        self.quality = nn.Linear(hidden_dim, 1)
        self.correction = nn.Linear(hidden_dim, prompt_dim)
        for layer in (self.existence, self.fallback, self.correction):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.quality.weight)
        nn.init.constant_(self.quality.bias, 2.)

    @classmethod
    def from_checkpoint(cls, directory):
        """Explicit compatibility with the first pilot's fixed +/-4 prior."""
        directory = Path(directory)
        config = json.loads((directory / 'CONFIG.json').read_text())
        state = torch.load(directory / 'EVIDENCE.pth', map_location='cpu', weights_only=True)
        legacy = 'native_prior_log_weight' not in state
        head = cls(config['language_dim'], native_prior_init=4. if legacy else .1)
        if legacy:
            state['native_prior_log_weight'] = head.native_prior_log_weight.detach().clone()
        head.load_state_dict(state, strict=True)
        if legacy:
            head.native_prior_log_weight.requires_grad_(False)
        return head

    def forward(self, image_features, semantic, native_accepted, prompt=None, candidate_logits=None,
                mode='spatial'):
        if mode not in ('spatial', 'global'):
            raise ValueError('unknown evidence control')
        if image_features.ndim != 4 or semantic.ndim != 2 or image_features.shape[0] != semantic.shape[0]:
            raise ValueError('batched semantic/image evidence required')
        if (prompt is None) != (candidate_logits is None):
            raise ValueError('candidate prompt and logits must be present together')
        query = self.query(semantic.float())
        features = image_features.float()
        if mode == 'global':
            features = features.mean((-2, -1), keepdim=True).expand_as(features)
        keys = self.image_norm(self.image(features).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        alignment = (keys * query[:, :, None, None]).sum(1, keepdim=True) / query.shape[-1]**.5 + self.map_bias
        weights = alignment.flatten(2).softmax(-1)
        evidence = (keys.flatten(2) * weights).sum(-1)
        global_feature = keys.mean((-2, -1))
        statistics = torch.cat([alignment.mean((-2, -1)), alignment.amax((-2, -1))], -1)
        global_state = self.global_fusion(torch.cat([query, evidence, global_feature, statistics], -1))
        # Native accept/reject is a prior, not an irreversible early exit.
        # There is NO candidate mask/prompt/quality input to this branch.
        prior = torch.where(native_accepted.bool().reshape(-1), 1., -1.).to(query)
        prior = prior * F.softplus(self.native_prior_log_weight)
        existence = prior + self.existence(global_state).squeeze(-1)
        if prompt is None:
            corrected = self.fallback(global_state)
            quality = None
        else:
            candidate = F.interpolate((candidate_logits.float() > .5).float(),
                                      size=keys.shape[-2:], mode='area')
            if mode == 'global':
                candidate = candidate.mean((-2, -1), keepdim=True).expand_as(candidate)
            foreground = (keys * candidate).sum((-2, -1)) / candidate.sum((-2, -1)).clamp_min(1e-6)
            agreement = (alignment.sigmoid() * candidate).sum((-2, -1)) / candidate.sum((-2, -1)).clamp_min(1e-6)
            area = candidate.mean((-2, -1))
            fused = self.candidate_fusion(torch.cat([query, evidence, foreground, self.prompt(prompt.float()),
                                                   agreement, area], -1))
            quality = self.quality(fused).squeeze(-1)
            scale = prompt.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
            # Poor candidate -> larger correction, NOT an absence decision.
            delta = .25 * scale * self.correction(fused).tanh()
            corrected = prompt + (1 - quality.sigmoid())[:, None] * delta
        return {'grounding_prompt': corrected, 'existence_logit': existence,
                'quality_logit': quality, 'alignment_logits': alignment}


def sam_feature_grid(states):
    features = states['current_vision_feats'][-1]
    height, width = states['feat_sizes'][-1]
    return features.permute(1, 2, 0).reshape(features.shape[1], features.shape[2], height, width).float()


@torch.inference_mode()
def cached_prediction(head, codec, states, inputs, shape, mode='spatial'):
    """No human target, target code, presence label or quality label input."""
    import numpy as np
    image_features = sam_feature_grid(states)
    mask = np.zeros(shape, dtype=bool)
    results = []
    candidates = inputs['candidates'] or [None]
    for candidate in candidates:
        options = {} if candidate is None else {key: candidate[key] for key in ('prompt', 'candidate_logits')}
        result = head(image_features, inputs['semantic'], inputs['native_accepted'], mode=mode, **options)
        results.append({'existence_probability': float(result['existence_logit'].sigmoid()),
                        'quality_probability': None if result['quality_logit'] is None
                        else float(result['quality_logit'].sigmoid())})
        if float(result['existence_logit'].sigmoid()) < .5:
            continue
        logits = codec.model.inject_language_embd(states, result['grounding_prompt'][:, None], nf_nobj=(1, 1))
        mask |= (F.interpolate(logits, size=shape, mode='bilinear')[0, 0] > .5).cpu().numpy()
    return mask, results

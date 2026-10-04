"""Bounded stage-two FTG on the jointly adapted identity representation.

The joint public-video FTG checkpoint improves its identity-only readout but
loses that gain when the original state token is injected. This stage retains
the adapted Qwen and identity projection, discards the old factorized module,
freezes the complete foundation, and learns a fresh zero-initialized state
residual whose norm is capped at two percent of the identity prompt.
"""

from mmengine.config import read_base

with read_base():
    from .anchored_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'bounded_ftg'
model['grounding_residual_ratio'] = 0.02
model['pretrained_pth'] = (
    ARTIFACT_ROOT
    + 'models/FTG-Qwen3-VL-4B-SAM3-video-pilot-repaired-training.pth'
)
model['pretrained_ignore_prefixes'] = ['factorized_grounding.']

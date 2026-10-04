"""Prompt-trust-region FTG on the frozen strong public foundation."""

from mmengine.config import read_base

with read_base():
    from .anchored_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'bounded_ftg'
model['grounding_residual_ratio'] = 0.02

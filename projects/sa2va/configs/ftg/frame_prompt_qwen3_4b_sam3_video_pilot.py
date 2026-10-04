"""Controlled frame-specific prompt baseline for the FTG temporal pilot."""

from mmengine.config import read_base

with read_base():
    from .ftg_qwen3_4b_sam3_video_pilot import *


# Everything else (checkpoint, seed, public data, LoRA, losses, steps, and
# frozen SAM3) is inherited unchanged. This variant emits only the
# identity-conditioned frame state, without the persistent identity token.
model['grounding_variant'] = 'frame_prompt'

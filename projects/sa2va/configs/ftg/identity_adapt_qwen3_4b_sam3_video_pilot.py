"""Public-video identity-adaptation baseline on Qwen3-VL-4B + SAM3.

This uses exactly the joint FTG pilot's public data, schedule, Qwen LoRA, text
projection, and frozen SAM3, but retains the published single identity prompt.
It is the clean stage-one foundation for the two-stage FTG comparison.
"""

from mmengine.config import read_base

with read_base():
    from .ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'identity_memory'

"""Foundation-preserving FTG pilot on public MeViS-v2 and Long-RVOS.

Unlike the first strong-foundation run, this recipe does not adapt Qwen, its
token embeddings, or the pretrained Qwen-to-SAM identity projection. It learns
only a target-aware dynamic residual whose output is zero at initialization, so
step zero exactly reproduces the public foundation.
"""

from mmengine.config import read_base

with read_base():
    from .ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'anchored_ftg'
model['freeze_foundation'] = True
model['mllm']['llm_lora'] = None

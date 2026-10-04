"""Stage-two FTG on the public-video-adapted identity foundation.

Stage one adapts Qwen and the existing identity projection on public MeViS-v2
and Long-RVOS data. This stage loads that stronger identity grounding, discards
the jointly trained factorized interface, freezes the complete foundation, and
learns a fresh zero-initialized identity-conditioned state residual.
"""

from mmengine.config import read_base

with read_base():
    from .anchored_ftg_qwen3_4b_sam3_video_pilot import *


model['pretrained_pth'] = (
    ARTIFACT_ROOT
    + 'models/IdentityAdapt-Qwen3-VL-4B-SAM3-video-pilot-training.pth'
)
model['pretrained_ignore_prefixes'] = ['factorized_grounding.']

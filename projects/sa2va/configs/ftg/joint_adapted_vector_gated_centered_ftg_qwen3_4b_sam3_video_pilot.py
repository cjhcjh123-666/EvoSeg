"""Channel-gated temporal FTG on the jointly adapted video foundation.

This final interface pilot implements the element-wise gate in the FTG
equation. A zero-initialized 256-D sigmoid gate modulates the temporally centered
state after direction normalization, retaining the identity-relative 2% norm
ceiling while allowing dimension-specific dynamic corrections.
"""

from mmengine.config import read_base

with read_base():
    from .joint_adapted_gated_centered_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'vector_gated_centered_ftg'
model['pretrained_overlay_pth'] = (
    ARTIFACT_ROOT
    + 'runs/ftg/20261005_joint_adapted_gated_centered_ftg_video_pilot/iter_440.pth'
)

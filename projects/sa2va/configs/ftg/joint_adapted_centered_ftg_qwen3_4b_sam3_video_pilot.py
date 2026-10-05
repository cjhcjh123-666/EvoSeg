"""Temporal-centered FTG on the jointly adapted public-video foundation.

This decisive pilot starts from the successfully trained bounded residual and
changes only the state definition: for every referred object, the mean
identity-conditioned observation is removed across sampled frames before the
state residual is produced.  The foundation remains frozen and the residual
is retrained on the same public MeViS-v2 + Long-RVOS data and schedule.
"""

from mmengine.config import read_base

with read_base():
    from .joint_adapted_bounded_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'centered_ftg'

# Model initialization first reconstructs the jointly adapted identity
# foundation and a fresh factorized module, then overlays the small, non-zero
# stage-two residual before DeepSpeed wraps the model. This gives centered
# state a stable initialization without resuming an optimizer or iteration.
model['pretrained_overlay_pth'] = (
    ARTIFACT_ROOT
    + 'runs/ftg/20261005_joint_adapted_bounded_ftg_video_pilot/iter_440.pth'
)

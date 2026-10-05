"""Gated temporal-centered FTG on the jointly adapted video foundation.

This stage fixes the normalized-state control's amplitude bug: the learned gate
is applied after direction normalization, so each frame can use anywhere from
zero to two percent of the persistent identity prompt norm. Both the visual
observation and produced state are centered across frames for each object.
"""

from mmengine.config import read_base

with read_base():
    from .joint_adapted_centered_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'gated_centered_ftg'
model['pretrained_overlay_pth'] = (
    ARTIFACT_ROOT
    + 'runs/ftg/20261005_joint_adapted_centered_ftg_video_pilot/iter_440.pth'
)

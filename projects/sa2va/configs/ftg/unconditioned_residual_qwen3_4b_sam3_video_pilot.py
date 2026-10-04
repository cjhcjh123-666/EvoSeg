"""Parameter-matched frame-residual control for foundation-preserving FTG.

This shares the frozen public foundation, zero-initialized residual branch,
data, and optimizer schedule with Anchored FTG, but state extraction receives
no persistent identity query.  It isolates identity-conditioned state from the
generic benefit of adding a frame-dependent residual.
"""

from mmengine.config import read_base

with read_base():
    from .anchored_ftg_qwen3_4b_sam3_video_pilot import *


model['grounding_variant'] = 'unconditioned_residual'

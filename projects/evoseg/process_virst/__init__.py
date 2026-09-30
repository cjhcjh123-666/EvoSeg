"""Process-aware VIRST research modules.

The package intentionally does not vendor or modify the official VIRST checkout.
Integration happens through explicit adapters around VIRST's public model modules.
"""

from .monotonic import MonotonicAlignment, MonotonicAlignmentOutput
from .process_module import ProcessConditioner, ProcessConditionerOutput

__all__ = [
    "MonotonicAlignment",
    "MonotonicAlignmentOutput",
    "ProcessConditioner",
    "ProcessConditionerOutput",
]

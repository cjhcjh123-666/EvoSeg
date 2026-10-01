"""QwenProcessSeg research utilities.

The package deliberately keeps the Qwen-to-SAM3.1 bridge separate from the
pixel decoder.  SAM3.1 remains the only component that produces mask logits.
"""


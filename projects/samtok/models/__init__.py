"""Native mask inference without importing optional training frameworks."""
from importlib import import_module

from .sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

_OPTIONAL_EXPORTS = {
    'PerceptionLM_TokenMask': '.perceptionlm',
    'QWEN25VL_VQSAM2Model': '.qwen25vl',
    'QWEN3VL_VQSAM2Model': '.qwen3vl',
    'PerceptionLMProcessor': '.processing_perception_lm',
    'VQ_SAM2Model': '.vq_sam2',
}


def __getattr__(name):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_OPTIONAL_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = ['VQ_SAM2', 'VQ_SAM2Config', 'SAM2Config', 'DirectResize', *_OPTIONAL_EXPORTS]

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

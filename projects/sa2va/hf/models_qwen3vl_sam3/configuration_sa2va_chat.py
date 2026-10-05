import copy

import transformers
from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

logger = logging.get_logger(__name__)

class Sa2VAChatConfigQwen(Qwen3VLConfig):
    model_type = 'sa2va_chat'

    def __init__(
            self,
            template=None,
            grounding_variant='identity_memory',
            grounding_residual_ratio=0.02,
            temporal_sampling='first',
            temporal_budget=5,
            identity_temporal_sampling=None,
            state_temporal_sampling=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.template = template
        self.grounding_variant = grounding_variant
        self.grounding_residual_ratio = grounding_residual_ratio
        self.temporal_sampling = temporal_sampling
        self.temporal_budget = temporal_budget
        self.identity_temporal_sampling = identity_temporal_sampling
        self.state_temporal_sampling = state_temporal_sampling

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """

        output = super().to_dict()
        output["template"] = self.template
        output["grounding_variant"] = self.grounding_variant
        output["grounding_residual_ratio"] = self.grounding_residual_ratio
        output["temporal_sampling"] = self.temporal_sampling
        output["temporal_budget"] = self.temporal_budget
        output["identity_temporal_sampling"] = self.identity_temporal_sampling
        output["state_temporal_sampling"] = self.state_temporal_sampling

        return output

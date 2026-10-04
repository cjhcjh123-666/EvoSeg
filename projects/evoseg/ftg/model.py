"""Qwen3-VL + Factorized Temporal Grounding + frozen official SAM3.1."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from torch import nn

from projects.evoseg.qwen_process_seg.baseline_model import QwenSegSAM31

from .interface import FactorizedTemporalGrounding


class FTGQwenSAM31(QwenSegSAM31):
    """Replace only the grounding interface of the controlled Qwen/SAM baseline."""

    def __init__(
        self,
        qwen_checkpoint: Path,
        sam_checkpoint: Path,
        sam_official_repo: Path,
        variant: str = "ftg",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        qwen_pixels: int = 128 * 28 * 28,
    ) -> None:
        super().__init__(
            qwen_checkpoint=qwen_checkpoint,
            sam_checkpoint=sam_checkpoint,
            sam_official_repo=sam_official_repo,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            qwen_pixels=qwen_pixels,
        )
        hidden = self.qwen.config.text_config.hidden_size
        prompt_dim = self.executor.prompt_dim
        self.fusion.requires_grad_(False)
        self.bridge.requires_grad_(False)
        self.grounding = FactorizedTemporalGrounding(
            hidden_dim=hidden,
            prompt_dim=prompt_dim,
            variant=variant,
        )
        self.variant = variant
        # A near-zero (rather than exactly zero) scale preserves the native
        # SAM3.1 text baseline while allowing gradients to reach Qwen/FTG on
        # the first optimization step.
        self.native_residual_scale = nn.Parameter(torch.tensor(1e-3))
        self.grounding.activate_variant_parameters()

    def encode_qwen(
        self, frames: Sequence[Image.Image], query: str
    ) -> tuple[torch.Tensor, dict]:
        frame_states, query_state, diagnostics = self.encode_qwen_context(frames, query)
        prompts, grounding_diagnostics = self.grounding(frame_states, query_state)
        if getattr(self, "sam_interface", "detector_grounding") == "native_text_residual":
            prompts = self.native_residual_scale * prompts
            grounding_diagnostics["native_residual_scale"] = self.native_residual_scale
        diagnostics.update(grounding_diagnostics)
        return prompts, diagnostics

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        qwen_lora = [
            parameter for name, parameter in self.qwen.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        grounding = [
            parameter for parameter in self.grounding.parameters()
            if parameter.requires_grad
        ]
        grounding.append(self.native_residual_scale)
        return {"qwen_lora": qwen_lora, "grounding": grounding}

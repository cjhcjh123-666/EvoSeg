"""Frozen official SAM3.1 pixel executor with a differentiable prompt path."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .bridge import multiplex_prompt_batch


def _tensor(value: object) -> torch.Tensor:
    return value.tensors if hasattr(value, "tensors") else value


class FrozenSAM31Executor(nn.Module):
    """Official SAM3.1 decoder; no custom pixel-prediction parameters."""

    def __init__(
        self,
        checkpoint: Path,
        official_repo: Path,
        visual_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if str(official_repo.resolve()) not in sys.path:
            sys.path.insert(0, str(official_repo.resolve()))
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(checkpoint),
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
        self.assembled = predictor.model
        self.assembled.requires_grad_(False)
        self.assembled.eval()
        self.tracker = self.assembled.tracker.model
        self.visual_chunk_size = visual_chunk_size
        self.image_size = int(self.assembled.image_size)
        self.prompt_dim = int(self.tracker.sam_mask_decoder.transformer_dim)
        self.load_report = report.getvalue()
        if "Missing keys (" in self.load_report or "Unexpected keys (" in self.load_report:
            raise RuntimeError("assembled SAM3.1 checkpoint did not load cleanly")

    @staticmethod
    def preprocess(frames: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
        tensors = []
        for frame in frames:
            array = np.asarray(frame.convert("RGB"), dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
            tensors.append(tensor)
        batch = torch.stack(tensors).to(device)
        batch = F.interpolate(
            batch,
            size=(1008, 1008),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (batch - 0.5) / 0.5

    def extract_visual_features(
        self, frames: Sequence[Image.Image], device: torch.device
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        images = self.preprocess(frames, device)
        embeddings = []
        high_zero = []
        high_one = []
        decoder = self.tracker.sam_mask_decoder
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk in images.split(self.visual_chunk_size):
                outputs = self.assembled.detector.backbone.forward_image(
                    chunk,
                    need_sam3_out=False,
                    need_interactive_out=False,
                    need_propagation_out=True,
                )
                pyramid = outputs["sam2_backbone_out"]["backbone_fpn"]
                high_zero.append(decoder.conv_s0(_tensor(pyramid[0])).detach())
                high_one.append(decoder.conv_s1(_tensor(pyramid[1])).detach())
                embeddings.append(_tensor(pyramid[-1]).detach())
        return torch.cat(embeddings), [torch.cat(high_zero), torch.cat(high_one)]

    def decode(
        self,
        frame_prompts: torch.Tensor,
        visual_features: tuple[torch.Tensor, list[torch.Tensor]],
    ) -> torch.Tensor:
        """Return one official predicted logit mask per frame, object slot 0."""
        image_embedding, high_res = visual_features
        extra = multiplex_prompt_batch(
            frame_prompts,
            self.tracker.output_valid_embed,
            self.tracker.output_invalid_embed,
            object_slot=0,
        )
        decoder = self.tracker.sam_mask_decoder
        decoder.eval()  # Frozen official decoder; multimask selection uses predicted IoU.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = decoder(
                image_embeddings=image_embedding,
                image_pe=self.tracker.get_propagation_dense_pe(),
                high_res_features=high_res,
                multimask_output=True,
                extra_per_object_embeddings=extra,
            )
        masks = output["masks"][:, 0]  # [T,K,H,W]
        quality = output["iou_pred"][:, 0]  # [T,K]
        chosen = quality.argmax(dim=-1)
        rows = torch.arange(masks.shape[0], device=masks.device)
        return masks[rows, chosen]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.assembled.parameters() if p.requires_grad)

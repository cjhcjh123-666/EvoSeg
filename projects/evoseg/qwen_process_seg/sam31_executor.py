"""Frozen official SAM3.1 pixel executor with a differentiable prompt path."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
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
        self.detector = self.assembled.detector
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

    def extract_grounding_features(
        self, frames: Sequence[Image.Image], device: torch.device
    ) -> dict[str, list[torch.Tensor]]:
        """Cache official SAM3.1 detector features, with no visual gradients."""
        images = self.preprocess(frames, device)
        fpn_by_level: list[list[torch.Tensor]] = []
        pos_by_level: list[list[torch.Tensor]] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk in images.split(self.visual_chunk_size):
                output = self.detector.backbone.forward_image(
                    chunk,
                    need_sam3_out=True,
                    need_interactive_out=False,
                    need_propagation_out=False,
                )
                if not fpn_by_level:
                    fpn_by_level = [[] for _ in output["backbone_fpn"]]
                    pos_by_level = [[] for _ in output["vision_pos_enc"]]
                for destination, feature in zip(fpn_by_level, output["backbone_fpn"]):
                    destination.append(_tensor(feature).detach())
                for destination, position in zip(pos_by_level, output["vision_pos_enc"]):
                    destination.append(_tensor(position).detach())
        return {
            "backbone_fpn": [torch.cat(level) for level in fpn_by_level],
            "vision_pos_enc": [torch.cat(level) for level in pos_by_level],
        }

    def decode_grounding_prompts(
        self,
        frame_prompts: torch.Tensor,
        grounding_features: dict[str, list[torch.Tensor]],
        decode_chunk_size: int | None = None,
        return_all_queries: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode learned prompt tokens through SAM3.1's official grounding path.

        The token enters ``Sam3Image._encode_prompt`` as the supported
        ``visual_prompt_embed`` and then traverses the official vision-language
        encoder, object-query decoder, and official segmentation head.  Query
        selection uses SAM3.1's own predicted grounding logit, never GT.
        """
        from sam3.model.geometry_encoders import Prompt

        if frame_prompts.ndim != 2 or frame_prompts.shape[-1] != self.prompt_dim:
            raise ValueError("frame prompts must be [T, official_prompt_dim]")
        if decode_chunk_size is None:
            decode_chunk_size = int(getattr(self, "decode_chunk_size", 4))
        if decode_chunk_size <= 0:
            raise ValueError("decode_chunk_size must be positive")
        total = frame_prompts.shape[0]
        backbone_out = {
            **grounding_features,
            "language_features": torch.zeros(
                0, total, self.prompt_dim,
                device=frame_prompts.device,
                dtype=frame_prompts.dtype,
            ),
            "language_mask": torch.zeros(
                total, 0, device=frame_prompts.device, dtype=torch.bool
            ),
        }
        predicted_masks = []
        predicted_scores = []
        self.detector.eval()
        for start in range(0, total, decode_chunk_size):
            end = min(start + decode_chunk_size, total)
            ids = torch.arange(start, end, device=frame_prompts.device)
            batch = end - start
            find_input = SimpleNamespace(img_ids=ids, text_ids=ids)
            geometry = Prompt(
                box_embeddings=torch.zeros(
                    0, batch, 4, device=frame_prompts.device, dtype=frame_prompts.dtype
                ),
                box_mask=torch.zeros(
                    batch, 0, device=frame_prompts.device, dtype=torch.bool
                ),
            )
            visual_prompt = frame_prompts[start:end].unsqueeze(0)
            visual_prompt_mask = torch.zeros(
                batch, 1, device=frame_prompts.device, dtype=torch.bool
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prompt, prompt_mask, current_backbone = self.detector._encode_prompt(
                    backbone_out,
                    find_input,
                    geometry,
                    visual_prompt_embed=visual_prompt,
                    visual_prompt_mask=visual_prompt_mask,
                    encode_text=False,
                )
                current_backbone, encoder_out, _ = self.detector._run_encoder(
                    current_backbone, find_input, prompt, prompt_mask
                )
                out = {
                    "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                    "prev_encoder_out": {
                        "encoder_out": encoder_out,
                        "backbone_out": current_backbone,
                    },
                }
                out, hidden = self.detector._run_decoder(
                    memory=out["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )
                self.detector._run_segmentation_heads(
                    out=out,
                    backbone_out=current_backbone,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hidden,
                )
            score = out["pred_logits"].squeeze(-1)
            if return_all_queries:
                predicted_masks.append(out["pred_masks"])
                predicted_scores.append(score)
                continue
            chosen = score.argmax(dim=-1)
            rows = torch.arange(batch, device=chosen.device)
            predicted_masks.append(out["pred_masks"][rows, chosen])
        masks = torch.cat(predicted_masks)
        if return_all_queries:
            return masks, torch.cat(predicted_scores)
        return masks

    def decode(
        self,
        frame_prompts: torch.Tensor,
        visual_features: tuple[torch.Tensor, list[torch.Tensor]],
        return_all_queries: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        if return_all_queries:
            return masks, quality
        chosen = quality.argmax(dim=-1)
        rows = torch.arange(masks.shape[0], device=masks.device)
        return masks[rows, chosen]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.assembled.parameters() if p.requires_grad)

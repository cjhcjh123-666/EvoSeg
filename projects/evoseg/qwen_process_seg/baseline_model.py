"""Frame-aware QwenSeg-SAM31 baseline with no process module."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .bridge import FrameQueryFusion, QwenToSAM31Bridge
from .frame_protocol import gather_frame_tokens, recover_frame_token_spans
from .qwen_frame_audit import find_subsequence
from .sam31_executor import FrozenSAM31Executor


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2 * (probability * target).flatten(1).sum(-1)
    denominator = probability.flatten(1).sum(-1) + target.flatten(1).sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


class QwenSegSAM31(nn.Module):
    def __init__(
        self,
        qwen_checkpoint: Path,
        sam_checkpoint: Path,
        sam_official_repo: Path,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        qwen_pixels: int = 128 * 28 * 28,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(qwen_checkpoint, trust_remote_code=True)
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        qwen.requires_grad_(False)
        last_six = range(qwen.config.text_config.num_hidden_layers - 6, qwen.config.text_config.num_hidden_layers)
        targets = [
            f"language_model.layers.{layer}.self_attn.{projection}"
            for layer in last_six
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        self.qwen = get_peft_model(qwen, config)
        hidden = qwen.config.text_config.hidden_size
        self.fusion = FrameQueryFusion(hidden)
        self.bridge = QwenToSAM31Bridge(hidden, 256)
        self.executor = FrozenSAM31Executor(sam_checkpoint, sam_official_repo)
        self.qwen_pixels = qwen_pixels

    def encode_qwen(
        self, frames: Sequence[Image.Image], query: str
    ) -> tuple[torch.Tensor, dict]:
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            min_pixels=self.qwen_pixels,
            max_pixels=self.qwen_pixels,
            padding=True,
            return_tensors="pt",
        ).to(next(self.qwen.parameters()).device)
        output = self.qwen(**inputs, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        spans = recover_frame_token_spans(
            inputs.input_ids,
            inputs.image_grid_thw,
            list(range(len(frames))),
            self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            self.processor.image_processor.merge_size,
        )
        _, summaries = gather_frame_tokens(hidden, spans)
        query_ids = self.processor.tokenizer(query, add_special_tokens=False).input_ids
        query_span = find_subsequence(inputs.input_ids[0].tolist(), query_ids)
        query_global = hidden[0, query_span[0] : query_span[1]].mean(dim=0)
        states = self.fusion(summaries, query_global)
        prompts = self.bridge(states)
        diagnostics = {
            "visual_token_count": sum(span.token_count for span in spans),
            "frame_token_spans": [(span.token_start, span.token_end) for span in spans],
            "prompt_cross_frame_std": prompts.float().std(dim=0).mean(),
        }
        return prompts, diagnostics

    def forward(
        self,
        frames: Sequence[Image.Image],
        query: str,
        masks: torch.Tensor,
        sam_visual_features: dict[str, list[torch.Tensor]] | None = None,
    ) -> dict:
        prompts, diagnostics = self.encode_qwen(frames, query)
        visual = sam_visual_features or self.executor.extract_grounding_features(
            frames, prompts.device
        )
        all_logits, scores = self.executor.decode_grounding_prompts(
            prompts, visual, return_all_queries=True
        )
        target = F.interpolate(
            masks[:, None].to(all_logits.device),
            size=all_logits.shape[-2:],
            mode="nearest",
        )[:, 0]
        expanded_target = target[:, None].expand_as(all_logits)
        per_query_bce = F.binary_cross_entropy_with_logits(
            all_logits.float(), expanded_target.float(), reduction="none"
        ).flatten(2).mean(-1)
        probability = all_logits.float().sigmoid()
        per_query_dice = 1 - (
            2 * (probability * expanded_target).flatten(2).sum(-1) + 1
        ) / (
            probability.flatten(2).sum(-1)
            + expanded_target.flatten(2).sum(-1)
            + 1
        )
        # Standard training-only bipartite assignment for one target per frame.
        # GT chooses the supervised official object query, never an inference prompt.
        matched_query = (per_query_bce + per_query_dice).detach().argmin(dim=-1)
        rows = torch.arange(all_logits.shape[0], device=all_logits.device)
        logits = all_logits[rows, matched_query]
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
        dice = soft_dice_loss(logits.float(), target.float())
        selection = F.cross_entropy(scores.float(), matched_query)
        predicted_query = scores.argmax(dim=-1)
        predicted_logits = all_logits[rows, predicted_query]
        predicted_bce = F.binary_cross_entropy_with_logits(
            predicted_logits.float(), target.float()
        )
        predicted_dice = soft_dice_loss(predicted_logits.float(), target.float())
        diagnostics.update(
            {
                "loss_bce": bce,
                "loss_dice": dice,
                "loss_selection": selection,
                "matched_mask_loss": bce + dice,
                "predicted_mask_loss": predicted_bce + predicted_dice,
                "mask_empty_fraction": (predicted_logits.sigmoid() < 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "mask_full_fraction": (predicted_logits.sigmoid() >= 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "matched_mask_logits": logits,
                "predicted_mask_logits": predicted_logits,
                "matched_query": matched_query,
                "predicted_query": predicted_query,
            }
        )
        diagnostics["loss"] = bce + dice + 0.1 * selection
        return diagnostics

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {"qwen_lora": [], "fusion": [], "bridge": []}
        groups["qwen_lora"] = [
            parameter for name, parameter in self.qwen.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        groups["fusion"] = list(self.fusion.parameters())
        groups["bridge"] = list(self.bridge.parameters())
        return groups

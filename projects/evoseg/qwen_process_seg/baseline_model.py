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

    def forward(self, frames: Sequence[Image.Image], query: str, masks: torch.Tensor) -> dict:
        prompts, diagnostics = self.encode_qwen(frames, query)
        visual = self.executor.extract_visual_features(frames, prompts.device)
        logits = self.executor.decode(prompts, visual)
        target = F.interpolate(
            masks[:, None].to(logits.device),
            size=logits.shape[-2:],
            mode="nearest",
        )[:, 0]
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
        dice = soft_dice_loss(logits.float(), target.float())
        diagnostics.update(
            {
                "loss_bce": bce,
                "loss_dice": dice,
                "mask_empty_fraction": (logits.sigmoid() < 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "mask_full_fraction": (logits.sigmoid() >= 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "mask_logits": logits,
            }
        )
        diagnostics["loss"] = bce + dice
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

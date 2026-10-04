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


def query_assignments(
    per_query_loss: torch.Tensor,
    scores: torch.Tensor,
    policy: str,
    fixed_query_index: int = 0,
    match_scope: str = "frame",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return oracle, supervised, and inference query indices per frame."""
    if per_query_loss.shape != scores.shape:
        raise ValueError("per-query loss and score tensors must have the same shape")
    if match_scope == "frame":
        oracle = per_query_loss.detach().argmin(dim=-1)
    elif match_scope == "video":
        clip_query = per_query_loss.detach().mean(dim=0).argmin()
        oracle = clip_query.expand(per_query_loss.shape[0])
    else:
        raise ValueError(f"unknown match scope: {match_scope}")
    if policy == "predicted_score":
        return oracle, oracle, scores.argmax(dim=-1)
    if policy == "fixed_slot":
        if not 0 <= fixed_query_index < scores.shape[-1]:
            raise ValueError("fixed query index is outside the SAM query bank")
        fixed = torch.full_like(oracle, fixed_query_index)
        return oracle, fixed, fixed
    raise ValueError(f"unknown query policy: {policy}")


def query_selection_loss(
    scores: torch.Tensor,
    matched_query: torch.Tensor,
    loss_type: str,
    positive_weight: float = 5.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Train SAM query objectness with either the legacy or official semantics.

    SAM3.1 emits one binary objectness logit per object query.  Its native
    training objective marks the Hungarian-matched query positive and every
    other query negative; a softmax over query indices is retained only as a
    controlled legacy ablation.
    """
    if scores.ndim != 2 or matched_query.shape != scores.shape[:1]:
        raise ValueError("scores must be [T,Q] and matched_query must be [T]")
    if loss_type == "softmax_ce":
        return F.cross_entropy(scores.float(), matched_query)
    if loss_type != "binary_objectness":
        raise ValueError(f"unknown selection loss: {loss_type}")

    targets = torch.zeros_like(scores, dtype=torch.float32)
    targets.scatter_(1, matched_query[:, None], 1.0)
    logits = scores.float()
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # Match SAM3.1's IABCE structure: emphasize the single positive query and
    # focal-downweight easy negatives without changing inference behavior.
    positive = targets * bce * positive_weight
    negative = (1.0 - targets) * bce * probability.pow(gamma)
    return (positive + negative).mean(dim=1).mean()


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
        self.query_policy = "predicted_score"
        self.fixed_query_index = 0
        self.selection_loss_type = "softmax_ce"
        self.match_scope = "frame"

    def encode_qwen_context(
        self, frames: Sequence[Image.Image], query: str
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
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
        diagnostics = {
            "visual_token_count": sum(span.token_count for span in spans),
            "frame_token_spans": [(span.token_start, span.token_end) for span in spans],
        }
        return summaries, query_global, diagnostics

    def encode_qwen(
        self, frames: Sequence[Image.Image], query: str
    ) -> tuple[torch.Tensor, dict]:
        summaries, query_global, diagnostics = self.encode_qwen_context(frames, query)
        states = self.fusion(summaries, query_global)
        prompts = self.bridge(states)
        diagnostics["prompt_cross_frame_std"] = prompts.float().std(dim=0).mean()
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
        matched_query, supervised_query, predicted_query = query_assignments(
            per_query_bce + per_query_dice,
            scores,
            self.query_policy,
            self.fixed_query_index,
            self.match_scope,
        )
        rows = torch.arange(all_logits.shape[0], device=all_logits.device)
        matched_logits = all_logits[rows, matched_query]
        logits = all_logits[rows, supervised_query]
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
        dice = soft_dice_loss(logits.float(), target.float())
        matched_bce = F.binary_cross_entropy_with_logits(
            matched_logits.float(), target.float()
        )
        matched_dice = soft_dice_loss(matched_logits.float(), target.float())
        selection = query_selection_loss(
            scores,
            matched_query,
            getattr(self, "selection_loss_type", "softmax_ce"),
        )
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
                "matched_mask_loss": matched_bce + matched_dice,
                "supervised_mask_loss": bce + dice,
                "predicted_mask_loss": predicted_bce + predicted_dice,
                "mask_empty_fraction": (predicted_logits.sigmoid() < 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "mask_full_fraction": (predicted_logits.sigmoid() >= 0.5).all(dim=-1).all(dim=-1).float().mean(),
                "matched_mask_logits": matched_logits,
                "predicted_mask_logits": predicted_logits,
                "matched_query": matched_query,
                "supervised_query": supervised_query,
                "predicted_query": predicted_query,
            }
        )
        selection_weight = (
            float(getattr(self, "selection_loss_weight", 0.1))
            if self.query_policy == "predicted_score" else 0.0
        )
        diagnostics["query_policy"] = self.query_policy
        diagnostics["selection_loss_type"] = self.selection_loss_type
        diagnostics["match_scope"] = self.match_scope
        diagnostics["selection_loss_weight"] = selection_weight
        diagnostics["loss"] = bce + dice + selection_weight * selection
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

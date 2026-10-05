"""Warm-started FTG on the official SAM 3.1 interactive prompt interface.

The Sa2VA export supplies *only* its trained Qwen and SEG projection. All
pixel features and mask heads come from the genuine SAM 3.1 checkpoint.
No SAM3 PVS parameters are silently substituted for SAM3.1 parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from projects.evoseg.qwen_process_seg.frame_protocol import gather_frame_tokens, recover_frame_token_spans
from projects.evoseg.qwen_process_seg.sam31_executor import FrozenSAM31Executor, _tensor


def load_trained_qwen(source: Path):
    raw = json.loads((source / "config.json").read_text())
    config = Qwen3VLConfig(
        text_config=raw["text_config"], vision_config=raw["vision_config"],
        image_token_id=raw["image_token_id"], video_token_id=raw["video_token_id"],
        vision_start_token_id=raw["vision_start_token_id"],
        vision_end_token_id=raw["vision_end_token_id"], tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    # Meta construction leaves non-persistent RoPE buffers unmaterialized.
    qwen = Qwen3VLForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    expected = set(qwen.state_dict())
    available = {key.removeprefix("model.") for key in index if key.startswith("model.")}
    if expected != available:
        raise RuntimeError({"missing_qwen": sorted(expected - available), "extra_qwen": sorted(available - expected)})
    loaded = set()
    projection = {}
    for shard in sorted(set(index.values())):
        relevant = [key for key, filename in index.items() if filename == shard and
                    (key.startswith("model.") or key.startswith("text_hidden_fcs."))]
        if not relevant:
            continue
        with safe_open(str(source / shard), framework="pt", device="cpu") as tensors:
            weights = {}
            for key in relevant:
                tensor = tensors.get_tensor(key)
                if key.startswith("model."):
                    name = key.removeprefix("model.")
                    weights[name] = tensor.to(torch.bfloat16)
                    loaded.add(name)
                else:
                    projection[key.removeprefix("text_hidden_fcs.")] = tensor.float()
            qwen.load_state_dict(weights, strict=False, assign=True)
    if loaded != expected or any(parameter.is_meta for parameter in qwen.parameters()):
        raise RuntimeError("incomplete pretrained Qwen initialization")
    hidden = config.text_config.hidden_size
    projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 256), nn.Dropout(0.0))
    projector.load_state_dict(projection, strict=True)
    return qwen, projector, {"qwen_tensors": len(loaded), "projection_tensors": len(projection), "source": str(source)}


class EvidenceFactorization(nn.Module):
    """One video identity; ordered, identity-conditioned local state.

    The identity reads the entire ordered clip once. The state subsequently
    reads each frame's spatial features, without independently choosing a new
    referent. This is a trainable implementation of the FTG hypothesis, not
    evidence that the hypothesis has already been validated.
    """
    def __init__(self, dim: int = 256, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True),
            num_layers=2, enable_nested_tensor=False,
        )
        self.identity_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.identity_update = nn.Linear(dim, dim)
        self.state_update = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gate = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.identity_update.weight, std=1e-3)
        nn.init.zeros_(self.identity_update.bias)
        nn.init.normal_(self.state_update[-1].weight, std=1e-3)
        nn.init.zeros_(self.state_update[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, identity, frame_semantics, spatial, times=None):
        length, dim = frame_semantics.shape
        if times is None:
            times = torch.linspace(0, 1, length, device=spatial.device)
        frequencies = torch.exp(torch.arange(0, dim, 2, device=spatial.device).float() * (-9.210340372 / dim))
        phase = times.float()[:, None] * 100 * frequencies[None]
        position = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1).to(frame_semantics.dtype)
        ordered = self.temporal((self.norm(frame_semantics) + position)[None])
        evidence, _ = self.identity_attention(self.norm(identity)[None, None], ordered, ordered, need_weights=False)
        persistent = identity + self.identity_update(evidence[0, 0])
        local_query = self.norm(ordered[0] + persistent)[..., None, :]
        spatial_evidence, _ = self.spatial_attention(local_query, self.norm(spatial), self.norm(spatial), need_weights=False)
        joined = torch.cat((persistent.expand(length, -1), ordered[0] + spatial_evidence[:, 0]), dim=-1)
        state = self.state_update(joined)
        gate = self.gate(joined).sigmoid()
        prompts = persistent[None] + gate * state
        return prompts, {"identity": persistent, "state": state, "gate_mean": gate.mean(), "state_norm": state.float().norm(dim=-1).mean()}


class PrimaryFTGSAM31(nn.Module):
    def __init__(self, foundation: Path, sam_checkpoint: Path, sam_repo: Path, qwen_pixels=200704, lora_rank=16):
        super().__init__()
        self.processor = Qwen3VLProcessor.from_pretrained(foundation)
        qwen, self.projector, self.initialization_report = load_trained_qwen(foundation)
        qwen.requires_grad_(False)
        layers = range(qwen.config.text_config.num_hidden_layers - 6, qwen.config.text_config.num_hidden_layers)
        targets = [f"language_model.layers.{layer}.self_attn.{name}" for layer in layers for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        self.qwen = get_peft_model(qwen, LoraConfig(r=lora_rank, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=targets, task_type="CAUSAL_LM"))
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.factorization = EvidenceFactorization()
        self.executor = FrozenSAM31Executor(sam_checkpoint, sam_repo, visual_chunk_size=1)
        self.qwen_pixels = qwen_pixels
        self.seg_token_id = self.processor.tokenizer.convert_tokens_to_ids("[SEG]")
        if self.seg_token_id == self.processor.tokenizer.unk_token_id:
            raise RuntimeError("foundation tokenizer has no SEG token")

    def train(self, mode=True):
        super().train(mode)
        self.executor.eval()
        return self

    def encode(self, frames, query):
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": f"Please segment {query}"})
        messages = [{"role": "user", "content": content}, {"role": "assistant", "content": "Sure, [SEG]."}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=frames, min_pixels=self.qwen_pixels, max_pixels=self.qwen_pixels, return_tensors="pt").to(next(self.qwen.parameters()).device)
        output = self.qwen(**inputs, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        selected = inputs.input_ids[0] == self.seg_token_id
        if selected.sum().item() != 1:
            raise RuntimeError("expected exactly one identity SEG token")
        spans = recover_frame_token_spans(inputs.input_ids, inputs.image_grid_thw, list(range(len(frames))), self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"), self.processor.image_processor.merge_size)
        _, summaries = gather_frame_tokens(hidden, spans)
        identity = self.projector(hidden[0, selected][0].float())
        frame_semantics = self.projector(summaries.float())
        return identity, frame_semantics

    def extract_visual(self, frames, device):
        embeddings, high0, high1 = [], [], []
        images = self.executor.preprocess(frames, device)
        tracker = self.executor.tracker
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in images.split(1):
                # The assembled predictor owns the shared vision backbone in
                # detector.backbone; tracker.backbone is intentionally None.
                shared = tracker.share_necks
                outputs = self.executor.detector.backbone.forward_image(
                    image, need_sam3_out=False,
                    need_interactive_out=not shared, need_propagation_out=shared,
                )
                pyramid = outputs["sam2_backbone_out" if shared else "interactive"]["backbone_fpn"]
                neck_decoder = tracker.sam_mask_decoder if shared else tracker.interactive_sam_mask_decoder
                embeddings.append(_tensor(pyramid[-1]).detach())
                high0.append(neck_decoder.conv_s0(_tensor(pyramid[0])).detach())
                high1.append(neck_decoder.conv_s1(_tensor(pyramid[1])).detach())
        return torch.cat(embeddings), [torch.cat(high0), torch.cat(high1)]

    def predict(self, frames, query, times=None):
        identity, semantics = self.encode(frames, query)
        embeddings, high = self.extract_visual(frames, identity.device)
        spatial = F.adaptive_avg_pool2d(embeddings.float(), (16, 16)).flatten(2).transpose(1, 2)
        prompts, diagnostics = self.factorization(identity, semantics, spatial, times)
        tracker = self.executor.tracker
        prompt_encoder = tracker.interactive_sam_prompt_encoder
        _, dense = prompt_encoder(points=None, boxes=None, masks=None)
        masks, scores = [], []
        # Each frame receives the same refined identity, plus its local state.
        # No GT mask, point, box, query index, or visibility enters this path.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(len(frames)):
                decoded, _, _, presence = tracker.interactive_sam_mask_decoder(
                    image_embeddings=embeddings[index:index+1] + tracker.interactivity_no_mem_embed.reshape(1, -1, 1, 1),
                    image_pe=prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=prompts[index:index+1, None],
                    dense_prompt_embeddings=dense,
                    multimask_output=False, repeat_image=False,
                    high_res_features=[feature[index:index+1] for feature in high],
                )
                masks.append(decoded[:, 0])
                scores.append(presence.reshape(-1))
        logits = torch.cat(masks)
        presence = torch.cat(scores)
        return logits, presence, diagnostics

    def forward(self, frames, query, masks, times=None):
        logits, presence, diagnostics = self.predict(frames, query, times)
        target = F.interpolate(masks[:, None].to(logits.device).float(), size=logits.shape[-2:], mode="nearest")[:, 0]
        probability = logits.float().sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits.float(), target)
        dice = (1 - (2 * (probability * target).flatten(1).sum(-1) + 1) / (probability.flatten(1).sum(-1) + target.flatten(1).sum(-1) + 1)).mean()
        visible = target.flatten(1).any(-1).float()
        visibility = F.binary_cross_entropy_with_logits(presence.float(), visible)
        return {"loss": bce + dice + 0.1 * visibility, "bce": bce, "dice": dice, "visibility": visibility,
                "predicted_presence_fraction": (presence > 0).float().mean(),
                "predicted_foreground_fraction": (logits > 0).float().mean(),
                "predicted_mask_logits": torch.where(presence[:, None, None] > 0, logits, -32.0), **diagnostics}

    def parameter_groups(self):
        return {"qwen_lora": [p for p in self.qwen.parameters() if p.requires_grad],
                "grounding": list(self.projector.parameters()) + list(self.factorization.parameters())}

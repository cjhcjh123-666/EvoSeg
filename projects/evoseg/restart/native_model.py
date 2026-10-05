"""Strict released weights and native-architecture adaptation, without old FTG."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .prepare_foundation import validate_inventory


def load_release(model_dir, assets_file, device):
    state = json.loads(Path(assets_file).read_text())
    if state['status'] != 'ASSETS_READY':
        raise RuntimeError('checkpoint download has not passed inventory checks')
    validate_inventory(Path(model_dir), state['weight_files'])
    model, loading = AutoModel.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, local_files_only=True,
        output_loading_info=True)
    bad = {key: loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
           if loading.get(key)}
    if bad:
        raise RuntimeError(f'incomplete native checkpoint loading: {bad}')
    index = json.loads((Path(model_dir) / 'model.safetensors.index.json').read_text())['weight_map']
    own = model.state_dict()
    if set(index) != set(own):
        raise RuntimeError(f'native state inventory differs: missing={sorted(set(own)-set(index))[:10]}, '
                           f'unexpected={sorted(set(index)-set(own))[:10]}')
    if model.config.samurai_mode:
        raise RuntimeError('native SAM2 required; SAMURAI is not this baseline')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    for token in ('[SEG]', '<IMG_CONTEXT>'):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] == tokenizer.unk_token_id:
            raise RuntimeError(f'published special token contract broken: {token}')
    model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
    return model.to(device).eval(), tokenizer, {'loading_info': loading,
        'revision': state['revision'], 'state_tensors': len(own), 'complete_loading': True}


def prepare_adaptation(model, lora_rank=32):
    # The published InternLM2 code hardcodes torch.checkpoint without kwargs.
    # Its reentrant default is incompatible with DDP's unused-parameter
    # detection. Change activation recomputation only, never weights/forward
    # semantics, in this dedicated process (no shared environment mutation).
    checkpoint_fn = torch.utils.checkpoint.checkpoint
    if not getattr(checkpoint_fn, '_native_non_reentrant', False):
        def checkpoint_non_reentrant(function, *values, **kwargs):
            kwargs.setdefault('use_reentrant', False)
            return checkpoint_fn(function, *values, **kwargs)
        checkpoint_non_reentrant._native_non_reentrant = True
        torch.utils.checkpoint.checkpoint = checkpoint_non_reentrant
    model.requires_grad_(False)
    model.wrap_llm_lora(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=.05)
    model.language_model.gradient_checkpointing_enable()
    model.language_model.enable_input_require_grads()
    model.language_model.config.use_cache = False
    model.text_hidden_fcs.requires_grad_(True)
    model.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
    # Accumulate optimizer state in float32 for newly added adapters and heads.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def restore_adaptation(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    prepare_adaptation(model, checkpoint['lora_rank'])
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if set(checkpoint['trainable_state']) != expected:
        raise RuntimeError('adaptation checkpoint trainable inventory differs')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint['trainable_state'].items():
            if parameters[name].shape != value.shape:
                raise RuntimeError(f'adaptation tensor shape differs: {name}')
            parameters[name].copy_(value)
    model.eval()
    return checkpoint


def point_sample(values, coordinates):
    return F.grid_sample(values, coordinates[:, :, None] * 2 - 1,
                         align_corners=False)[:, :, :, 0]


def native_mask_losses(logits, targets, num_points=12544):
    """Official point sampling (3x, 75% uncertainty), CE x2 and naive Dice x.5."""
    logits = logits.float()[:, None]
    targets = F.interpolate(targets.float()[:, None], size=logits.shape[-2:], mode='nearest')
    with torch.no_grad():
        candidates = torch.rand(len(logits), num_points * 3, 2, device=logits.device)
        uncertainties = -point_sample(logits, candidates)[:, 0].abs()
        indices = uncertainties.topk(int(num_points * .75), dim=1).indices
        selected = candidates.gather(1, indices[:, :, None].expand(-1, -1, 2))
        random_points = torch.rand(len(logits), num_points - selected.shape[1], 2, device=logits.device)
        coordinates = torch.cat([selected, random_points], dim=1)
        truth = point_sample(targets, coordinates)[:, 0]
    predictions = point_sample(logits, coordinates)[:, 0]
    ce = 2 * F.binary_cross_entropy_with_logits(predictions, truth)
    prob = predictions.sigmoid()
    dice = .5 * (1 - (2 * (prob * truth).sum(1) + 1) /
                 (prob.sum(1) + truth.sum(1) + 1)).mean()
    return ce, dice


class NativeFineTune(nn.Module):
    def __init__(self, foundation):
        super().__init__()
        self.foundation = foundation

    def train(self, mode=True):
        super().train(mode)
        # Frozen visual encoders are deterministic; learned decoder and LoRA train.
        self.foundation.vision_model.eval()
        self.foundation.mlp1.eval()
        return self

    def forward(self, sample):
        model = self.foundation
        device = next(model.parameters()).device
        ids = sample['input_ids'].to(device)[None]
        labels = sample['labels'].to(device)[None]
        pixels = sample['pixel_values'].to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            # Same native feature extraction, split into image minibatches only.
            vision = torch.cat([model.extract_feature(chunk) for chunk in pixels.split(2)], dim=0)
        base_lm = model.language_model.get_base_model()
        embedded = base_lm.get_input_embeddings()(ids).clone()
        selected = ids == model.img_context_token_id
        if int(selected.sum()) != vision.shape[0] * vision.shape[1]:
            raise RuntimeError('native image embedding count differs from prompt')
        embedded[selected] = vision.reshape(-1, vision.shape[-1]).to(embedded)
        # Equivalent LM forward; evaluate its vocabulary head only where labels
        # are supervised, avoiding a 5k x 92k unused prompt-logit allocation.
        output = base_lm.model(inputs_embeds=embedded, attention_mask=torch.ones_like(ids),
                               position_ids=torch.arange(ids.shape[1], device=device)[None],
                               use_cache=False, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state
        supervised = labels[:, 1:] != -100
        logits = base_lm.get_output_embeddings()(hidden[:, :-1][supervised])
        llm_loss = F.cross_entropy(logits.float(), labels[:, 1:][supervised])
        seg = model.text_hidden_fcs(hidden[ids == model.seg_token_idx])
        targets = sample['targets'].to(device)
        frame_count, objects = targets.shape[:2]
        if seg.shape[0] != frame_count * objects:
            raise RuntimeError('native object-major SEG and frame-major masks disagree')
        prompts = seg.reshape(objects, frame_count, -1).permute(1, 0, 2)
        sam = model.grounding_encoder.sam2_model
        mask_ce, mask_dice = [], []
        # Original training is independent keyframes, no synthetic memory chain.
        for frame_index, image in enumerate(sample['grounding_pixels']):
            image = model.grounding_encoder.preprocess_image(image.to(device), dtype=torch.float32)
            with torch.no_grad():
                features = sam.image_encoder(image[None].to(torch.bfloat16))
            # forward_image projects high-res features using learned decoder
            # convs. Keep those outside no_grad to preserve their gradients.
            if sam.use_high_res_features_in_sam:
                features['backbone_fpn'][0] = sam.sam_mask_decoder.conv_s0(features['backbone_fpn'][0])
                features['backbone_fpn'][1] = sam.sam_mask_decoder.conv_s1(features['backbone_fpn'][1])
            _, backbone, _, sizes = sam._prepare_backbone_features(features)
            high_res = [x.permute(1, 2, 0).reshape(1, x.shape[2], *size).expand(objects, -1, -1, -1)
                        for x, size in zip(backbone[:-1], sizes[:-1])]
            image_embeddings = (backbone[-1] + sam.no_mem_embed).permute(1, 2, 0).reshape(
                1, sam.hidden_dim, *sizes[-1]).expand(objects, -1, -1, -1)
            result = sam._forward_sam_heads(
                backbone_features=image_embeddings, high_res_features=high_res,
                point_inputs=None, mask_inputs=None,
                multimask_output=sam._use_multimask(is_init_cond_frame=True, point_inputs=None),
                language_embd=prompts[frame_index][:, None])
            ce, dice = native_mask_losses(result[3][:, 0], targets[frame_index])
            mask_ce.append(ce)
            mask_dice.append(dice)
        losses = {'llm': llm_loss, 'mask_ce': torch.stack(mask_ce).mean(),
                  'mask_dice': torch.stack(mask_dice).mean()}
        return sum(losses.values()), {name: value.detach() for name, value in losses.items()}

"""One actual public-data gradient/update probe; not a benchmark model result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import decode_annotation
from .scope_head import TypedScopeHead


def run(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'PROBE_STATE.pth').exists():
        raise RuntimeError('probe already exists; do not overwrite its evidence')
    cache = Path(args.cache)
    manifest = json.loads((cache / 'MANIFEST.json').read_text())
    if manifest['generated_drafts_included'] or not manifest['validation_images_excluded']:
        raise RuntimeError('training probe requires public train-only cache')
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('incomplete public language checkpoint loading')
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'))
    if any('visual' in name and value.requires_grad for name, value in model.named_parameters()):
        raise RuntimeError('visual foundation unexpectedly trainable')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = model.cuda().train()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    width = model.config.text_config.hidden_size
    head = TypedScopeHead(width, width).cuda().train()
    record = next(json.loads(path.read_text()) for path in sorted((cache / 'refcoco').glob('*.json'))
                  if len(json.loads(path.read_text())['compiled_turns']) >= 2)
    target_turn = record['compiled_turns'][1]
    if target_turn['private_supervision']['witness_rounds'] != [1]:
        raise RuntimeError('selected probe round does not have the expected public witness reference')
    with Image.open(record['image']) as original:
        image = original.convert('RGB')
    messages = []
    for turn in record['compiled_turns'][:2]:
        content = [{'type': 'text', 'text': turn['query']}]
        if turn['round'] == 1:
            content.insert(0, {'type': 'image', 'image': image})
        messages.append({'role': 'user', 'content': content})
        if turn['round'] == 1:
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': turn['assistant_target']}]})
    prompt = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt')
    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': target_turn['assistant_target']}]})
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                          return_dict=True, return_tensors='pt').to('cuda')
    prefix = prompt.input_ids.shape[1]
    if not torch.equal(inputs.input_ids[0, :prefix].cpu(), prompt.input_ids[0]):
        raise RuntimeError('native teacher-forcing prefix differs from generation input')
    labels = inputs.input_ids.clone()
    labels[:, :prefix] = -100
    image_features = torch.load(record['image_feature_cache'], map_location='cpu', weights_only=True)[None].cuda()
    annotations = {int(a['id']): a for a in json.loads(Path(args.annotations).read_text())['annotations']}
    anno = annotations[target_turn['private_supervision']['target_annotation_id']]
    gt = decode_annotation(anno, image.height, image.width)
    target = torch.from_numpy(gt.astype(np.float32))[None, None].cuda()
    target = F.interpolate(target, size=image_features.shape[-2:], mode='nearest')[:, 0]
    previous_codes = record['compiled_turns'][0]['private_supervision']['target_codes']
    if previous_codes is None:
        raise RuntimeError('probe requires a valid historical target code pair')
    token_ids = [processor.tokenizer.convert_tokens_to_ids(f'<|mt_{code+depth*256:04d}|>')
                 for depth, code in enumerate(previous_codes)]
    trainable = [value for value in model.parameters() if value.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-6)
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        result = model(**inputs, labels=labels, output_hidden_states=True, use_cache=False)
        # Query representation is BEFORE the current GT assistant output.
        # Causal attention prevents this auxiliary route from reading its target.
        query = result.hidden_states[-1][:, prefix - 1]
        history = model.get_input_embeddings()(torch.tensor([token_ids], device='cuda')).mean(1)[:, None]
        predicted = head(query, history, torch.ones(1, 1, dtype=torch.bool, device='cuda'), image_features)
        pointer_loss = F.cross_entropy(predicted['role_pointer_logits'].float().reshape(2, 2),
                                       torch.tensor([1, 0], device='cuda'))
        operation_loss = F.cross_entropy(predicted['operation_logits'].float(), torch.tensor([0], device='cuda'))
        scope_loss = F.binary_cross_entropy_with_logits(predicted['scope_logits'].float(), target)
        loss = result.loss + .1 * pointer_loss + .01 * operation_loss + .1 * scope_loss
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite training probe loss')
    loss.backward()
    groups = {'language_lora': 0., 'history_role_router': 0., 'spatial_scope': 0.}
    for name, value in model.named_parameters():
        if value.requires_grad and value.grad is not None:
            if not torch.isfinite(value.grad).all():
                raise RuntimeError('non-finite language adapter gradient')
            groups['language_lora'] += float(value.grad.abs().sum())
    for name, value in head.named_parameters():
        if value.grad is not None:
            if not torch.isfinite(value.grad).all():
                raise RuntimeError('non-finite scope adapter gradient')
            if name.startswith(('role_queries', 'history', 'null_keys')):
                groups['history_role_router'] += float(value.grad.abs().sum())
            if name.startswith(('pixel_', 'scope_bias')):
                groups['spatial_scope'] += float(value.grad.abs().sum())
    if any(value <= 0 for value in groups.values()):
        raise RuntimeError(f'broken trainable gradient route: {groups}')
    torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    optimizer.step()
    state = {'language_adapters': {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad},
             'scope_head': {name: value.detach().cpu() for name, value in head.state_dict().items()},
             'steps': 1, 'purpose': 'gradient/update probe only, not method benchmark checkpoint'}
    torch.save(state, output / 'PROBE_STATE.pth')
    report = {'status': 'TRAINING_GRADIENT_PROBE_PASS', 'optimizer_updates': 1,
              'source': 'public_train_only', 'editing_branch_supervised': False,
              'gradient_l1_before_clip': groups, 'query_before_target_position': prefix - 1,
              'losses': {'total': float(loss), 'language': float(result.loss),
                         'witness_pointer': float(pointer_loss), 'operation': float(operation_loss),
                         'scope': float(scope_loss)}, 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
              'sota_claimed': False, 'benchmark_complete': False,
              'next_gate': 'controlled native inference integration and short public-train pilot'}
    atomic_json(output / 'PROBE_REPORT.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--cache', default=str(base / 'results/evoseg_dialog_20261006/public_train_pilot_cache'))
    parser.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    parser.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    parser.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/training_probe'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()

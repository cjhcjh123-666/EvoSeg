"""Matched public LoRA vs auxiliary witness/scope alignment, NOT full SCCS."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.evoseg.restart.prepare_foundation import atomic_json
from .pilot_data import DATASETS, SPLIT_VERSION, cache_records
from .protocol import decode_annotation
from .scope_head import TypedScopeHead


class PublicAlignment(nn.Module):
    def __init__(self, model, auxiliary):
        super().__init__()
        self.language = model
        self.head = TypedScopeHead(model.config.text_config.hidden_size,
                                  model.config.text_config.hidden_size) if auxiliary else None

    def forward(self, inputs, labels, prefix, history_ids, history_valid, pixels, target, pointer):
        result = self.language(**inputs, labels=labels, use_cache=False,
                               output_hidden_states=self.head is not None)
        losses = {'language': result.loss}
        if self.head is not None:
            # Strictly before the current teacher-forced assistant answer.
            query = result.hidden_states[-1][:, prefix - 1]
            history = self.language.get_input_embeddings()(history_ids).mean(2)
            outputs = self.head(query, history, history_valid, pixels)
            n = history_ids.shape[1] + 1
            losses['witness'] = F.cross_entropy(outputs['role_pointer_logits'].float().reshape(2, n), pointer)
            losses['operation'] = F.cross_entropy(outputs['operation_logits'].float(),
                                                  torch.zeros(1, dtype=torch.long, device=query.device))
            losses['scope'] = F.binary_cross_entropy_with_logits(outputs['scope_logits'].float(), target)
            total = result.loss + .1 * losses['witness'] + .01 * losses['operation'] + .1 * losses['scope']
        else:
            total = result.loss
        return total, {key: value.detach() for key, value in losses.items()}


def prepare(record, round_index, processor, annotations, auxiliary):
    current = record['compiled_turns'][round_index]
    if current['private_supervision']['edit_rounds'] or current['private_supervision']['operation'] != 'new':
        raise RuntimeError('public witness stage cannot silently claim editing supervision')
    with Image.open(record['image']) as source:
        image = source.convert('RGB')
    messages = []
    for index, turn in enumerate(record['compiled_turns'][:round_index + 1]):
        content = [{'type': 'text', 'text': turn['query']}]
        if index == 0:
            content.insert(0, {'type': 'image', 'image': image})
        messages.append({'role': 'user', 'content': content})
        if index < round_index:
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': turn['assistant_target']}]})
    prompt = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt')
    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': current['assistant_target']}]})
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                           return_dict=True, return_tensors='pt')
    prefix = prompt.input_ids.shape[1]
    if not torch.equal(inputs.input_ids[0, :prefix], prompt.input_ids[0]):
        raise RuntimeError('teacher forcing prefix mismatch')
    if inputs.input_ids.shape[1] > 12000:
        raise RuntimeError('unexpectedly long public pilot sequence; do not skip silently')
    inputs = inputs.to('cuda')
    labels = inputs.input_ids.clone()
    labels[:, :prefix] = -100
    if not auxiliary:
        return inputs, labels, prefix, None, None, None, None, None
    ids, valid = [], []
    for turn in record['compiled_turns'][:round_index]:
        codes = turn['private_supervision']['target_codes']
        valid.append(codes is not None)
        codes = codes if codes is not None else [0, 0]
        ids.append([processor.tokenizer.convert_tokens_to_ids(f'<|mt_{code+depth*256:04d}|>')
                    for depth, code in enumerate(codes)])
    history = torch.tensor(ids, dtype=torch.long, device='cuda').reshape(1, round_index, 2)
    validity = torch.tensor([valid], dtype=torch.bool, device='cuda')
    references = current['private_supervision']['witness_rounds']
    if len(references) > 1:
        raise RuntimeError('single-pointer public pilot cannot supervise multiple witnesses')
    witness = references[0] if references else 0
    if witness > round_index or (witness and not valid[witness - 1]):
        raise RuntimeError('invalid historical witness supervision')
    pointer = torch.tensor([witness, 0], dtype=torch.long, device='cuda')
    pixels = torch.load(record['image_feature_cache'], map_location='cpu', weights_only=True)[None].cuda()
    annotation = annotations[current['private_supervision']['target_annotation_id']]
    image_id = int(Path(record['image']).stem.split('_')[-1])
    mask = decode_annotation(annotation, image.height, image.width, image_id)
    target = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
    target = F.interpolate(target, size=pixels.shape[-2:], mode='nearest')[:, 0]
    return inputs, labels, prefix, history, validity, pixels, target, pointer


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'adapter').exists():
        raise RuntimeError('existing pilot evidence cannot be overwritten')
    records = cache_records(args.cache)
    heldout = cache_records(args.cache, holdout=True)
    assert not ({r['image'] for r in records} & {r['image'] for r in heldout})
    by_dataset = {d: [r for r in records if r['dataset'] == d] for d in DATASETS}
    auxiliary = args.variant == 'witness_scope_aux'
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir,
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('incomplete native language checkpoint')
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'))
    if any('visual' in n and p.requires_grad for n, p in model.named_parameters()):
        raise RuntimeError('pixel encoder must remain frozen')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    module = PublicAlignment(model, auxiliary).cuda().train()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    annotations = ({int(a['id']): a for a in json.loads(Path(args.annotations).read_text())['annotations']}
                   if auxiliary else None)
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    ddp = (torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank], find_unused_parameters=auxiliary)
           if world > 1 else module)
    manifest = {'variant': args.variant, 'split': SPLIT_VERSION, 'seed': args.seed,
                'train_dialogues': len(records), 'holdout_dialogues': len(heldout),
                'train_images': sorted({r['image'] for r in records}),
                'holdout_images': sorted({r['image'] for r in heldout}),
                'generated_drafts_included': False, 'editing_branch_supervised': False,
                'full_sccs_inference': False, 'aux_head_used_at_inference': False,
                'inference_effect': 'trained language LoRA only; scope head is auxiliary alignment',
                'world_size': world, 'accumulation': args.accumulation,
                'effective_batch': world * args.accumulation, 'updates': args.updates,
                'lora_rank': 8, 'learning_rate': args.lr, 'max_supervised_round': 6,
                'foundation_revision': records[0]['foundation_revision'],
                'training_started': True, 'sota_claimed': False}
    if rank == 0:
        atomic_json(output / 'TRAINING_MANIFEST.json', manifest)
    started = time.time()
    sample_trace = hashlib.sha256()
    for step in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(1, device='cuda')
        details = {}
        for micro in range(args.accumulation):
            global_slot = ((step - 1) * args.accumulation + micro) * world + rank
            rng = random.Random(args.seed + global_slot)
            dataset = DATASETS[global_slot % len(DATASETS)]
            record = rng.choice(by_dataset[dataset])
            turn_index = rng.randrange(min(6, len(record['compiled_turns'])))
            sample_trace.update(f'{dataset}:{record["source_index"]}:{turn_index}\n'.encode())
            prepared = prepare(record, turn_index, processor, annotations, auxiliary)
            sync = ddp.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    loss, parts = ddp(*prepared)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite public pilot loss')
                (loss / args.accumulation).backward()
            loss_sum += loss.detach() / args.accumulation
            for key, value in parts.items():
                details[key] = details.get(key, 0.) + float(value) / args.accumulation
            del prepared, loss, parts
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
            loss_sum /= world
        elapsed = time.time() - started
        state = {'status': 'TRAINING', 'variant': args.variant, 'rank': rank, 'pid': os.getpid(),
                 'optimizer_updates': step, 'planned_updates': args.updates, 'loss': float(loss_sum),
                 'rank_losses': details, 'elapsed_seconds': elapsed,
                 'eta_seconds': elapsed / step * (args.updates - step),
                 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
                 'sample_trace_sha256': sample_trace.hexdigest(), 'updated_at_unix': time.time()}
        atomic_json(output / f'STATUS_rank{rank}.json', state)
        if rank == 0:
            print(json.dumps(state), flush=True)
    if rank == 0:
        module.language.save_pretrained(output / 'adapter')
        if module.head is not None:
            torch.save(module.head.state_dict(), output / 'AUX_SCOPE_HEAD.pth')
        manifest.update(status='PUBLIC_PILOT_TRAINING_COMPLETE', optimizer_updates=args.updates,
                        training_elapsed_seconds=time.time() - started)
        atomic_json(output / 'TRAINING_MANIFEST.json', manifest)
    if world > 1:
        dist.barrier()
    state.update(status='TRAINING_COMPLETE')
    atomic_json(output / f'STATUS_rank{rank}.json', state)
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--cache', default=str(base / 'results/evoseg_dialog_20261006/public_train_pilot_cache'))
    p.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    p.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    p.add_argument('--variant', choices=['plain_lora', 'witness_scope_aux'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=64)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.updates < 1 or args.accumulation < 1 or args.lr <= 0:
        p.error('positive training budget required')
    run(args)


if __name__ == '__main__':
    main()

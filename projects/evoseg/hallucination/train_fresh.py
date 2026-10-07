"""Fresh public SAMTok + inference-coupled grounding adapter, with matched controls."""
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
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.evoseg.restart.prepare_foundation import atomic_json
from projects.evoseg.dialog.runtime import NativeRuntime
from .interaction import InteractionGroundingAdapter


def user_message(image, query):
    return {'role': 'user', 'content': [{'type': 'image', 'image': image},
            {'type': 'text', 'text': f'Please segment {query} in this image.'}]}


def prefix_inputs(processor, image, query):
    return processor.apply_chat_template([user_message(image, query)], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt')


def teacher_inputs(processor, image, query, answer):
    prompt = prefix_inputs(processor, image, query)
    messages = [user_message(image, query), {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
    full = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt')
    prefix = prompt.input_ids.shape[1]
    if not torch.equal(full.input_ids[0, :prefix], prompt.input_ids[0]):
        raise RuntimeError('generation and training prefixes differ')
    labels = full.input_ids.clone()
    labels[:, :prefix] = -100
    return full, labels, prefix


def cached_records(cache):
    return [json.loads(path.read_text()) for path in sorted((Path(cache) / 'records').glob('*.json'))]


class CoupledGrounding(nn.Module):
    def __init__(self, language, codec, variant):
        super().__init__()
        self.language = language
        self.codec = codec.requires_grad_(False).eval()
        self.interface = InteractionGroundingAdapter(language.config.text_config.hidden_size)
        self.variant = variant

    def forward(self, full_inputs, labels, prefix, neutral_inputs, sam_states, target_codes, gt, empty,
                rollout_inputs=None, valid_pixels=None):
        # Optional proposal-input curriculum: predictions are INPUTS only.
        # The query, CE target, presence label and pixel supervision remain the
        # original public annotations. No model prediction becomes a label.
        prompt_codes = target_codes
        if rollout_inputs is not None:
            from .eval_fresh import parse_native_codes
            training = self.language.training
            self.language.eval()
            try:
                with torch.no_grad():
                    generated = self.language.generate(**rollout_inputs, max_new_tokens=128,
                        do_sample=False, use_cache=True)
                answer_ids = generated[0, rollout_inputs.input_ids.shape[1]:].tolist()
                answer = self.rollout_processor.decode(answer_ids, skip_special_tokens=False)
                proposals, _ = parse_native_codes(answer)
                # One prompt is corrected toward the human union-mask target.
                # Multiple proposals are deterministically sampled, not chosen
                # using mask overlap or GT. Empty/invalid output uses zero base.
                prompt_codes = proposals[self.rollout_index % len(proposals)] if proposals else None
            finally:
                self.language.train(training)
        output = self.language(**full_inputs, labels=labels, output_hidden_states=True, use_cache=False)
        full = output.hidden_states[-1][:, prefix - 1]
        controls = [self.language(**value, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states[-1][:, -1]
                    for value in neutral_inputs]
        visual, query, neither = controls
        if self.variant == 'full_view':
            # Equal forward budget; zero-weight controls leave full features
            # intact and make the gradient structure explicit in this control.
            visual, query, neither = [value * 0. for value in controls]
        if prompt_codes is None:
            base = torch.zeros((1, 256), device=full.device)
        else:
            values = torch.tensor([prompt_codes], device=full.device)
            embedding = self.codec.quantizer.embed_code(values)[:, None]
            base = self.codec.deconcate_quant_embed(embedding).reshape(1, -1)
            if base.shape[-1] != 256:
                raise RuntimeError('this first interface requires the released single 256-dim SAM prompt')
        result = self.interface(base, full, visual, query, neither)
        prompt = result['grounding_prompt'][:, None]
        mask_logits = self.codec.model.inject_language_embd(sam_states, prompt, nf_nobj=(1, 1))
        resized = F.interpolate(gt, size=mask_logits.shape[-2:], mode='nearest')
        valid = torch.ones_like(resized) if valid_pixels is None else F.interpolate(
            valid_pixels, size=mask_logits.shape[-2:], mode='nearest')
        if not valid.any():
            raise RuntimeError('no supervised pixels remain after ignore-region handling')
        mask_loss = (F.binary_cross_entropy_with_logits(mask_logits.float(), resized,
                     reduction='none') * valid).sum() / valid.sum()
        probability = mask_logits.float().sigmoid()
        dice = 1 - (2 * (probability * resized * valid).sum() + 1) / (
            (probability * valid).sum() + (resized * valid).sum() + 1)
        empty_loss = F.binary_cross_entropy_with_logits(result['empty_logit'].float(),
                                                       torch.tensor([float(empty)], device=full.device))
        loss = output.loss + .1 * empty_loss + .1 * mask_loss + .05 * dice
        return loss, {'language': output.loss.detach(), 'empty': empty_loss.detach(),
                      'mask': mask_loss.detach(), 'dice': dice.detach()}


def prepare(record, processor, rollout=False):
    with Image.open(record['image']) as original:
        image = original.convert('RGB')
    full, labels, prefix = teacher_inputs(processor, image, record['query'], record['assistant_target'])
    neutral_image = Image.new('RGB', image.size, (127, 127, 127))
    controls = [prefix_inputs(processor, image, 'an object'),
                prefix_inputs(processor, neutral_image, record['query']),
                prefix_inputs(processor, neutral_image, 'an object')]
    if not torch.equal(full.image_grid_thw, controls[1].image_grid_thw):
        raise RuntimeError('neutral image changed token geometry')
    states = torch.load(record['sam_states'], map_location=f'cuda:{torch.cuda.current_device()}', weights_only=True)
    with Image.open(record['human_gt_mask']) as mask:
        array = np.asarray(mask.convert('L')) > 0
    if bool(array.any()) == record['private_no_target']:
        raise RuntimeError('cached human mask/no-target label mismatch')
    gt = torch.from_numpy(array.astype(np.float32))[None, None].cuda()
    valid_pixels = None
    if record.get('human_valid_mask'):
        with Image.open(record['human_valid_mask']) as valid_image:
            valid = np.asarray(valid_image.convert('L')) > 0
        if valid.shape != array.shape or not valid.any():
            raise RuntimeError('invalid public ignore-region mask')
        valid_pixels = torch.from_numpy(valid.astype(np.float32))[None, None].cuda()
    return (full.to('cuda'), labels.cuda(), prefix, [value.to('cuda') for value in controls], states,
            record['private_target_codes'], gt, record['private_no_target'],
            prefix_inputs(processor, image, record['query']).to('cuda') if rollout else None, valid_pixels)


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [r for r in cached_records(args.cache) if not r['holdout']]
    if not records or any(r['generated_query'] or r['pseudo_label'] or r['source_split'] != 'train' for r in records):
        raise RuntimeError('requires original public TRAIN cache')
    buckets = {kind: [r for r in records if r['bucket'] == kind]
               for kind in ('ref_positive', 'gref_positive', 'gref_empty', 'reason_positive')}
    source_digest = records[0]['source_manifest_sha256']
    if any(r['source_manifest_sha256'] != source_digest for r in records):
        raise RuntimeError('mixed data manifests in training cache')
    language, loading = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('incomplete released language checkpoint')
    language.requires_grad_(False)
    parent = args.resume or args.init_from
    if parent:
        parent_config = json.loads((Path(parent) / 'CONFIG.json').read_text())
        if parent_config['variant'] != args.variant:
            raise RuntimeError('cannot initialize from a different interface variant')
        language = PeftModel.from_pretrained(language, Path(parent) / 'adapter',
                                             is_trainable=True, torch_device='cpu')
    else:
        language = get_peft_model(language, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    if any('visual' in name and parameter.requires_grad for name, parameter in language.named_parameters()):
        raise RuntimeError('visual foundation must stay frozen')
    language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    language.enable_input_require_grads()
    codec = NativeRuntime(args.model_dir, args.assets, load_language=False).tokenizer
    module = CoupledGrounding(language, codec, args.variant).cuda().train()
    module.codec.eval()
    if parent:
        module.interface.load_state_dict(torch.load(Path(parent) / 'GROUNDING.pth',
                                                    map_location='cpu', weights_only=True), strict=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    module.rollout_processor = processor
    module.rollout_index = 0
    language_parameters = [p for p in module.language.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': language_parameters, 'lr': args.lr},
                                  {'params': module.interface.parameters(), 'lr': args.head_lr}], weight_decay=.01)
    parameter_names = [n for n, p in module.named_parameters() if p.requires_grad]
    start = 0
    if args.resume:
        saved = torch.load(Path(args.resume) / 'TRAINING_STATE.pth', map_location='cpu', weights_only=True)
        if (saved['parameter_names'] != parameter_names or saved['variant'] != args.variant or
            saved['effective_batch'] != world * args.accumulation or saved['source_manifest_sha256'] != source_digest or
            saved.get('prompt_rollout_rate', 0.) != args.prompt_rollout_rate):
            raise RuntimeError('cannot resume different method/optimizer/data budget')
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
    model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False) if world > 1 else module
    trainable = [p for p in module.parameters() if p.requires_grad]
    started = time.time()
    trace = hashlib.sha256()

    def save(step):
        if rank == 0:
            directory = output / f'checkpoint_{step:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            module.language.save_pretrained(directory / 'adapter')
            torch.save(module.interface.state_dict(), directory / 'GROUNDING.pth')
            temporary = directory / 'STATE.tmp'
            torch.save({'optimizer': optimizer.state_dict(), 'parameter_names': parameter_names,
                        'variant': args.variant, 'step': step, 'effective_batch': world * args.accumulation,
                        'source_manifest_sha256': source_digest,
                        'prompt_rollout_rate': args.prompt_rollout_rate}, temporary)
            temporary.replace(directory / 'TRAINING_STATE.pth')
            atomic_json(directory / 'CONFIG.json', {'variant': args.variant, 'step': step,
                'language_dim': module.language.config.text_config.hidden_size, 'prompt_dim': 256,
                'head_used_at_inference': True, 'cache': args.cache, 'model_dir': args.model_dir,
                'old_faithful_inherited': False, 'generated_data': False, 'empty_threshold': .5,
                'initialization_checkpoint': args.init_from,
                'prompt_rollout_rate': args.prompt_rollout_rate,
                'proposal_predictions_are_inputs_not_labels': True})
            atomic_json(output / 'LATEST.json', {'checkpoint': str(directory), 'step': step})
        if world > 1:
            dist.barrier()

    for step in range(start + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(1, device='cuda')
        parts_sum = {}
        for micro in range(args.accumulation):
            slot = ((step - 1) * args.accumulation + micro) * world + rank
            rng = random.Random(42 + slot)
            kind = ('ref_positive', 'gref_positive', 'ref_positive', 'gref_empty')[slot % 4]
            if buckets['reason_positive'] and slot % 20 in (0, 9):
                # 10% published implicit-query supervision; maintain 25% of
                # official no-target examples, and do not paraphrase queries.
                kind = 'reason_positive'
            record = rng.choice(buckets[kind])
            trace.update(record['id'].encode())
            # Separate RNG stream preserves the same sampled original records
            # across the rollout curriculum and teacher-only training control.
            rollout = random.Random(1042 + slot).random() < args.prompt_rollout_rate
            module.rollout_index = slot
            prepared = prepare(record, processor, rollout=rollout)
            sync = model.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    loss, parts = model(*prepared)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite coupled training loss')
                (loss / args.accumulation).backward()
            loss_sum += loss.detach() / args.accumulation
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.) + float(value) / args.accumulation
            del prepared, loss, parts
        gradient = {key: float(sum(p.grad.float().abs().sum() for p in group if p.grad is not None))
                    for key, group in [('language', language_parameters), ('prompt', module.interface.prompt_residual.parameters()),
                                       ('empty', module.interface.empty_head.parameters())]}
        if any(not np.isfinite(v) or v <= 0 for v in gradient.values()):
            raise RuntimeError('broken gradient route: ' + str(gradient))
        torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
            loss_sum /= world
        atomic_json(output / f'STATUS_rank{rank}.json', {'status': 'TRAINING', 'pid': os.getpid(),
                    'variant': args.variant, 'step': step, 'planned_steps': args.steps, 'loss': float(loss_sum),
                    'loss_parts': parts_sum, 'gradient_l1': gradient, 'sample_trace_sha256': trace.hexdigest(),
                    'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9, 'updated_at_unix': time.time(),
                    'elapsed_seconds': time.time() - started, 'head_used_at_inference': True,
                    'prompt_rollout_rate': args.prompt_rollout_rate})
        if rank == 0:
            print(json.dumps({'step': step, 'variant': args.variant, 'loss': float(loss_sum), 'gradient': gradient}), flush=True)
        if step % args.save_every == 0 or step == args.steps or time.time() >= args.stop_at:
            save(step)
        if time.time() >= args.stop_at:
            break
    atomic_json(output / f'DONE_rank{rank}.json', {'step': step, 'status': 'TRAINING_COMPLETE'})
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    p.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    p.add_argument('--variant', choices=['full_view', 'interaction'], required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--head-lr', type=float, default=5e-5)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume')
    p.add_argument('--init-from', help='fork weights into a new objective/data experiment; reset optimizer and step')
    p.add_argument('--prompt-rollout-rate', type=float, default=0.,
                   help='fraction of original samples decoded from model proposal INPUTS, never pseudo labels')
    p.add_argument('--stop-at', type=float, default=float('inf'))
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if not 0 <= args.prompt_rollout_rate <= 1:
        p.error('prompt-rollout-rate must be in [0, 1]')
    run(args)


if __name__ == '__main__':
    main()

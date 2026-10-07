"""Cache frozen native proposals and train only a mask-conditioned grounding adapter."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from projects.evoseg.restart.prepare_foundation import atomic_json
from projects.evoseg.dialog.runtime import NativeRuntime
from .train_fresh import cached_records
from .eval_fresh import FreshRuntime, metric_values
from .region_evidence import RegionEvidenceAdapter, predict_regions
from .protocol import digest_file


FEATURE_KEYS = ('prompt', 'semantic', 'interaction', 'foreground', 'background', 'global_feature', 'area')


def select_records(source, maximum, limit=None):
    heldout = [r for r in source if r['holdout']]
    buckets = {kind: [r for r in source if not r['holdout'] and r['bucket'] == kind]
               for kind in ('ref_positive', 'gref_positive', 'gref_empty', 'reason_positive')}
    selected = []
    for kind, fraction in [('ref_positive', .4375), ('gref_positive', .21875),
                           ('gref_empty', .25), ('reason_positive', .09375)]:
        desired = int(maximum * fraction)
        if len(buckets[kind]) < desired:
            raise RuntimeError('not enough audited public source records: ' + kind)
        selected.extend(buckets[kind][:desired])
    rows = sorted(selected + heldout, key=lambda r: r['id'])
    return rows if limit is None else rows[:limit]


def cache(args):
    local = int(os.environ.get('LOCAL_RANK', 0))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = FreshRuntime(args.model_dir, args.assets, args.parent)
    if runtime.regional is not None:
        raise RuntimeError('regional pilot parent must be the preserved nonregional checkpoint')
    runtime.model.requires_grad_(False).eval()
    rows = select_records(cached_records(args.source_cache), args.max_records, args.limit)
    index = {'records': rows, 'parent': args.parent, 'parent_model_identity': runtime.adapter_fingerprint,
             'source_cache': args.source_cache, 'max_training_records': args.max_records,
             'smoke_only': args.limit is not None, 'original_queries_unchanged': True,
             'pseudo_labels': False, 'supervision': 'original human masks, ignore regions and official no-target labels'}
    if rank == 0:
        atomic_json(output / 'INDEX.json', index)
    started = time.time()
    for number, record in enumerate(rows[rank::world], 1):
        directory = output / 'records'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (record['id'] + '.json')
        source_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['source_record_sha256'] != source_hash or old['parent_model_identity'] != runtime.adapter_fingerprint:
                raise RuntimeError('stale regional proposal cache')
        else:
            # This call has no human mask, target ID, codes or presence input.
            prediction, information, proposals = predict_regions(runtime, record['image'], record['query'])
            with Image.open(record['human_gt_mask']) as image:
                target = np.asarray(image.convert('L')) > 0
            if record.get('human_valid_mask'):
                with Image.open(record['human_valid_mask']) as image:
                    valid = np.asarray(image.convert('L')) > 0
            else:
                valid = np.ones_like(target, dtype=bool)
            tensors = []
            for proposal in proposals:
                native = F.interpolate(proposal['parent_logits'], size=target.shape, mode='bilinear')[0, 0]
                candidate = (native > .5).cpu().numpy()
                precision = float((candidate & target & valid).sum() / max(1, (candidate & valid).sum()))
                # Soft foreground precision from the ORIGINAL GT; predictions
                # determine inputs, not mask/query/presence labels. Precision
                # keeps valid single-instance proposals in multi-target queries.
                tensors.append({**{k: v.detach().cpu().float() for k, v in proposal.items()},
                                'human_target_precision': torch.tensor([precision])})
            feature_path = output / 'features' / (record['id'] + '.pth')
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensors, feature_path)
            mask_path = output / 'parent_masks' / (record['id'] + '.png')
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
            atomic_json(destination, {**record, 'feature_path': str(feature_path),
                'parent_prediction': str(mask_path), 'native_proposals': len(proposals),
                'parent_inference': information, 'source_record_sha256': source_hash,
                'parent_model_identity': runtime.adapter_fingerprint, 'features_constructed_without_gt': True})
        atomic_json(output / f'STATUS_rank{rank}.json', {'status': 'CACHING_FROZEN_NATIVE_PROPOSALS',
            'records_done': number, 'records_total': len(rows[rank::world]), 'pid': os.getpid(),
            'elapsed_seconds': time.time() - started, 'updated_at_unix': time.time()})
    atomic_json(output / f'DONE_rank{rank}.json', {'records': len(rows[rank::world])})


class RegionTraining(nn.Module):
    def __init__(self, codec, language_dim, mode):
        super().__init__()
        self.codec = codec.requires_grad_(False).eval()
        self.head = RegionEvidenceAdapter(language_dim)
        self.mode = mode

    def forward(self, features, states, target, valid, empty):
        result = self.head(**{k: features[k] for k in FEATURE_KEYS}, mode=self.mode)
        logits = self.codec.model.inject_language_embd(states, result['grounding_prompt'][:, None], nf_nobj=(1, 1)).float()
        gt = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        weight = F.interpolate(valid, size=logits.shape[-2:], mode='nearest')
        if not weight.any():
            raise RuntimeError('no valid supervised pixels')
        mask_loss = (F.binary_cross_entropy_with_logits(logits, gt, reduction='none') * weight).sum() / weight.sum()
        probability = logits.sigmoid()
        dice = 1 - (2 * (probability * gt * weight).sum() + 1) / ((probability * weight).sum() + (gt * weight).sum() + 1)
        support_loss = F.binary_cross_entropy_with_logits(result['support_logit'], features['human_target_precision'])
        preservation = (logits - features['parent_logits']).square().mean() if not empty else logits.new_zeros(())
        loss = .1 * mask_loss + .05 * dice + .1 * support_loss + .02 * preservation
        return loss, {'mask': float(mask_loss.detach()), 'dice': float(dice.detach()),
                      'support': float(support_loss.detach()), 'positive_preservation': float(preservation.detach())}


def train(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.manual_seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(args.cache) / 'INDEX.json').read_text())
    if index['smoke_only'] and not args.allow_smoke:
        raise RuntimeError('smoke cache cannot enter the actual regional experiment')
    rows = [json.loads(p.read_text()) for p in sorted((Path(args.cache) / 'records').glob('*.json'))]
    if len(rows) != len(index['records']):
        raise RuntimeError('incomplete frozen proposal cache')
    eligible = [r for r in rows if not r['holdout'] and r['native_proposals']]
    if any(r['source_split'] != 'train' or r['generated_query'] or r['pseudo_label'] for r in eligible):
        raise RuntimeError('requires original public TRAIN supervision')
    if not eligible:
        raise RuntimeError('no original train proposal inputs')
    groups = {name: [r for r in eligible if r['bucket'] == name]
              for name in ('ref_positive', 'gref_positive', 'gref_empty', 'reason_positive')}
    codec = NativeRuntime(args.model_dir, args.assets, load_language=False).tokenizer
    config = json.loads((Path(index['parent']) / 'CONFIG.json').read_text())
    module = RegionTraining(codec, config['language_dim'], args.mode).cuda().train()
    module.codec.eval()
    model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False) if world > 1 else module
    optimizer = torch.optim.AdamW(module.head.parameters(), lr=args.lr, weight_decay=.01)
    started = time.time()

    def save(step):
        if rank == 0:
            destination = output / f'checkpoint_{step:06d}'
            destination.mkdir(parents=True, exist_ok=True)
            # Same frozen parent for both variants. Do not merge/update LoRA.
            shutil.copytree(Path(index['parent']) / 'adapter', destination / 'adapter', dirs_exist_ok=True)
            shutil.copyfile(Path(index['parent']) / 'GROUNDING.pth', destination / 'GROUNDING.pth')
            torch.save(module.head.state_dict(), destination / 'REGION.pth')
            atomic_json(destination / 'CONFIG.json', {**config, 'region_evidence_mode': args.mode,
                'regional_training_step': step, 'region_parent': index['parent'], 'language_frozen': True,
                'native_proposal_inputs_only': True, 'human_supervision_unchanged': True})
            torch.save({'optimizer': optimizer.state_dict(), 'step': step,
                        'cache_index_sha256': digest_file(Path(args.cache) / 'INDEX.json')}, destination / 'REGION_TRAINING_STATE.pth')
            atomic_json(output / 'LATEST.json', {'checkpoint': str(destination), 'step': step})
        if world > 1:
            dist.barrier()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for micro in range(2):
            slot = ((step - 1) * 2 + micro) * world + rank
            rng = random.Random(42 + slot)
            bucket = ('ref_positive', 'gref_positive', 'ref_positive', 'gref_empty')[slot % 4]
            if slot % 20 in (0, 9):
                bucket = 'reason_positive'
            available = groups[bucket] or eligible
            record = rng.choice(available)
            device = f'cuda:{torch.cuda.current_device()}'
            proposals = torch.load(record['feature_path'], map_location=device, weights_only=True)
            features = proposals[slot % len(proposals)]  # never selected by GT overlap
            states = torch.load(record['sam_states'], map_location=device, weights_only=True)
            with Image.open(record['human_gt_mask']) as mask:
                target = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None, None].cuda()
            if record.get('human_valid_mask'):
                with Image.open(record['human_valid_mask']) as mask:
                    valid = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None, None].cuda()
            else:
                valid = torch.ones_like(target)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                loss, parts = model(features, states, target, valid, record['private_no_target'])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite regional loss')
            (loss / 2).backward()
            total += float(loss.detach()) / 2
        gradients = {name: float(sum(p.grad.float().abs().sum() for p in layer.parameters() if p.grad is not None))
                     for name, layer in [('residual', module.head.residual), ('support', module.head.support)]}
        if any(not np.isfinite(v) or v <= 0 for v in gradients.values()):
            raise RuntimeError('broken regional gradient: ' + str(gradients))
        torch.nn.utils.clip_grad_norm_(module.head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        atomic_json(output / f'STATUS_rank{rank}.json', {'status': 'REGION_TRAINING', 'pid': os.getpid(),
            'step': step, 'planned_steps': args.steps, 'loss': total, 'loss_parts': parts,
            'gradient_l1': gradients, 'language_frozen': True, 'mode': args.mode,
            'eligible_train_records': len(eligible), 'eligible_buckets': dict(Counter(r['bucket'] for r in eligible)),
            'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9, 'updated_at_unix': time.time(),
            'elapsed_seconds': time.time() - started})
        if step % 100 == 0 or step == args.steps or time.time() >= args.stop_at:
            save(step)
        if time.time() >= args.stop_at:
            break
    atomic_json(output / f'DONE_rank{rank}.json', {'step': step})
    if world > 1:
        dist.destroy_process_group()


@torch.inference_mode()
def cached_prediction(proposals, shape, head, codec, states, mode, empty_threshold=.5):
    """Private diagnostic equivalent of native inference; no GT/label inputs."""
    mask = np.zeros(shape, dtype=bool)
    for proposal in proposals:
        if float(proposal['parent_empty_logit'].sigmoid()) >= empty_threshold:
            continue
        result = head(**{k: proposal[k] for k in FEATURE_KEYS}, mode=mode)
        if float(result['support_logit'].sigmoid()) < .5:
            continue
        logits = codec.model.inject_language_embd(states, result['grounding_prompt'][:, None], nf_nobj=(1, 1))
        mask |= (F.interpolate(logits, size=shape, mode='bilinear')[0, 0] > .5).cpu().numpy()
    return mask


def evaluate_cache(args):
    local = int(os.environ.get('LOCAL_RANK', 0))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted((Path(args.cache) / 'records').glob('*.json'))]
    rows = [r for r in rows if r['holdout']]
    codec = head = None
    if args.checkpoint:
        directory = Path(args.checkpoint)
        config = json.loads((directory / 'CONFIG.json').read_text())
        head = RegionEvidenceAdapter(config['language_dim']).cuda().eval()
        head.load_state_dict(torch.load(directory / 'REGION.pth', map_location='cpu', weights_only=True), strict=True)
        codec = NativeRuntime(args.model_dir, args.assets, load_language=False).tokenizer
        identity = digest_file(directory / 'REGION.pth') + ':' + config['region_evidence_mode']
    else:
        identity = json.loads((Path(args.cache) / 'INDEX.json').read_text())['parent_model_identity']
    started = time.time()
    for number, record in enumerate(rows[rank::world], 1):
        if head is None:
            with Image.open(record['parent_prediction']) as image:
                prediction = np.asarray(image) > 0
        else:
            with Image.open(record['parent_prediction']) as image:
                shape = (image.height, image.width)
            proposals = torch.load(record['feature_path'], map_location=device, weights_only=True)
            states = torch.load(record['sam_states'], map_location=device, weights_only=True)
            prediction = cached_prediction(proposals, shape, head, codec, states,
                                          config['region_evidence_mode'], config['empty_threshold'])
        with Image.open(record['human_gt_mask']) as image:
            target = np.asarray(image) > 0
        if record.get('human_valid_mask'):
            with Image.open(record['human_valid_mask']) as image:
                valid = np.asarray(image) > 0
        else:
            valid = None
        metrics = metric_values(target, prediction, record['parent_inference']['malformed_codes'], valid)
        destination = output / 'cases' / (record['id'] + '.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, {'id': record['id'], 'model_identity': identity, 'metrics': metrics,
                                  'private_frozen_feature_diagnostic': True})
        atomic_json(output / f'STATUS_rank{rank}.json', {'cases_done': number, 'updated_at_unix': time.time(),
                                                       'elapsed_seconds': time.time() - started})


def summarize_cache(args):
    output = Path(args.output)
    expected = [r for r in json.loads((Path(args.cache) / 'INDEX.json').read_text())['records'] if r['holdout']]
    rows = [json.loads((output / 'cases' / (r['id'] + '.json')).read_text()) for r in expected]
    if len({r['model_identity'] for r in rows}) != 1:
        raise RuntimeError('mixed cached diagnostic heads')
    values = [r['metrics'] for r in rows]
    positive, negative = [v for v in values if not v['empty_gt']], [v for v in values if v['empty_gt']]
    if not positive or not negative:
        raise RuntimeError('diagnostic needs target-present and absent samples')
    atomic_json(output / 'METRICS.json', {'cases': len(values), 'full_public_evaluation': False,
        'private_frozen_feature_diagnostic': True, 'sota_claimed': False, 'checkpoint': args.checkpoint,
        'gIoU': sum(v['iou'] for v in values) / len(values),
        'positive_gIoU': sum(v['iou'] for v in positive) / len(positive),
        'cIoU': sum(v['intersection'] for v in values) / max(1, sum(v['union'] for v in values)),
        'N_acc': sum(v['predicted_empty'] for v in negative) / len(negative),
        'T_acc': sum(not v['predicted_empty'] for v in positive) / len(positive)})


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['cache', 'train', 'evaluate-cache', 'summarize-cache'])
    p.add_argument('--model-dir', default=base + '/models/Qwen3-VL-8B-SAMTok-official')
    p.add_argument('--assets', default=base + '/results/evoseg_dialog_20261006/ASSETS.json')
    p.add_argument('--source-cache', default=base + '/results/evoseg_hallucination_20261007/reasoning_public_mix/cache')
    p.add_argument('--parent', default=base + '/results/evoseg_hallucination_20261007/new_public_run/training/interaction/checkpoint_000100')
    p.add_argument('--cache')
    p.add_argument('--output', required=True)
    p.add_argument('--max-records', type=int, default=1024)
    p.add_argument('--limit', type=int)
    p.add_argument('--mode', choices=['regional', 'global'], default='regional')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--stop-at', type=float, default=float('inf'))
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--checkpoint')
    args = p.parse_args()
    {'cache': cache, 'train': train, 'evaluate-cache': evaluate_cache,
     'summarize-cache': summarize_cache}[args.stage](args)


if __name__ == '__main__':
    main()

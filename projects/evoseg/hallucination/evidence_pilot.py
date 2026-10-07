"""Small matched evidence-routing pilot; no automatic public evaluation.

Reuses audited public TRAIN images/queries and frozen parent proposals as
inputs, never as labels. Human target/ignore masks supervise alignment,
presence, candidate precision and segmentation. Native-empty examples are
included. Frozen Qwen is needed only to obtain missing query states.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from projects.evoseg.restart.prepare_foundation import atomic_json
from .continue_training import preservation_guard
from .eval_fresh import FreshRuntime, metric_values
from .evidence_routing import EvidenceRoutingAdapter, cached_prediction, sam_feature_grid
from .protocol import digest_file, read_mask
from .run_read_alignment import select_records as select_holdout


def select_training(rows, limit):
    kinds = ('ref_positive', 'gref_positive', 'gref_empty', 'reason_positive')
    groups = {kind: sorted([r for r in rows if not r['holdout'] and r['bucket'] == kind],
                          key=lambda r: (kind == 'gref_empty' and not r['native_proposals'], r['id']))
              for kind in kinds}
    if limit == 1024:
        quotas = dict(zip(kinds, (448, 224, 256, 96)))
    else:
        if limit % 4:
            raise ValueError('balanced four-bucket train budget unavailable')
        quotas = {kind: limit // 4 for kind in kinds}
    if any(len(groups[kind]) < quotas[kind] for kind in kinds):
        raise ValueError('audited four-bucket train budget unavailable')
    result = [r for kind in kinds for r in groups[kind][:quotas[kind]]]
    if any(r['source_split'] != 'train' or r['generated_query'] or r['pseudo_label'] for r in result):
        raise ValueError('only unchanged audited public TRAIN records permitted')
    return result


def pack_inputs(proposals, semantic, threshold):
    accepted = [float(p['parent_empty_logit'].sigmoid()) < threshold for p in proposals]
    if len(set(accepted)) > 1:
        raise ValueError('parent existence prior must not vary across candidates')
    return {'semantic': semantic.detach().cpu().float(),
            'native_accepted': torch.tensor([any(accepted)]),
            'candidates': [{'prompt': p['prompt'].detach().cpu().float(),
                            'candidate_logits': p['parent_logits'].detach().cpu().float()} for p in proposals]}


def to_device(value, device='cuda'):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def targets(record, device='cuda'):
    gt = torch.from_numpy(read_mask(record['human_gt_mask']).astype(np.float32))[None, None].to(device)
    valid = torch.from_numpy(read_mask(record['human_valid_mask']).astype(np.float32))[None, None].to(device) \
            if record.get('human_valid_mask') else torch.ones_like(gt)
    if not valid.any() or bool((gt * valid).any()) == record['private_no_target']:
        raise ValueError('human mask/presence/ignore audit mismatch')
    return gt, valid


def supervised_loss(head, codec, states, inputs, target, valid, mode, step):
    candidates = inputs['candidates'] or [None]
    candidate = candidates[(step - 1) % len(candidates)]
    options = {} if candidate is None else {'prompt': candidate['prompt'], 'candidate_logits': candidate['candidate_logits']}
    result = head(sam_feature_grid(states), inputs['semantic'], inputs['native_accepted'], mode=mode, **options)
    shape = result['alignment_logits'].shape[-2:]
    weight = F.interpolate(valid, size=shape, mode='area')
    soft_gt = F.interpolate(target * valid, size=shape, mode='area') / weight.clamp_min(1e-6)
    positive_mass = (soft_gt * weight).sum()
    positive_weight = ((weight * (1 - soft_gt)).sum() / positive_mass.clamp_min(1)).clamp(1, 20)
    spatial_bce = (F.binary_cross_entropy_with_logits(result['alignment_logits'], soft_gt,
                   pos_weight=positive_weight, reduction='none') * weight).sum() / weight.sum()
    spatial_probability = result['alignment_logits'].sigmoid()
    spatial_dice = 1 - (2 * (spatial_probability * soft_gt * weight).sum() + 1) / (
        (spatial_probability * weight).sum() + positive_mass + 1)
    spatial = .5 * spatial_bce + .5 * spatial_dice
    present = (target * valid).flatten(1).sum(-1).gt(0).float()
    existence = F.binary_cross_entropy_with_logits(result['existence_logit'], present)
    quality = spatial.new_zeros(())
    precision = None
    if candidate is not None:
        native = F.interpolate(candidate['candidate_logits'].float(), size=target.shape[-2:], mode='bilinear') > .5
        precision = (native * target * valid).sum().reshape(1) / (native * valid).sum().clamp_min(1)
        quality = F.binary_cross_entropy_with_logits(result['quality_logit'], precision)
    loss = spatial + existence + .1 * quality
    pixel = spatial.new_zeros(())
    dice = spatial.new_zeros(())
    preserve = spatial.new_zeros(())
    # First learn correspondence; do not move segmentation prompts immediately.
    if step > 25:
        logits = codec.model.inject_language_embd(states, result['grounding_prompt'][:, None], nf_nobj=(1, 1)).float()
        gt = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        mask_valid = F.interpolate(valid, size=logits.shape[-2:], mode='nearest')
        pixel = (F.binary_cross_entropy_with_logits(logits, gt, reduction='none') * mask_valid).sum() / mask_valid.sum()
        probability = logits.sigmoid()
        dice = 1 - (2 * (probability * gt * mask_valid).sum() + 1) / ((probability * mask_valid).sum() + (gt * mask_valid).sum() + 1)
        if candidate is not None and present.item() and precision.item() > .8:
            preserve = (logits - candidate['candidate_logits'].float()).square().mean()
        loss = loss + .1 * pixel + .05 * dice + .02 * preserve
    return loss, {'alignment': float(spatial.detach()), 'existence': float(existence.detach()),
                  'quality': float(quality.detach()), 'pixel': float(pixel.detach()),
                  'dice': float(dice.detach()), 'positive_preservation': float(preserve.detach())}


def score_cases(cases):
    positive = [c for c in cases.values() if not c['empty_gt']]
    absent = [c for c in cases.values() if c['empty_gt']]
    return {'cases': len(cases), 'gIoU': sum(c['iou'] for c in cases.values()) / len(cases),
            'positive_gIoU': sum(c['iou'] for c in positive) / len(positive),
            'N_acc': sum(c['predicted_empty'] for c in absent) / len(absent),
            'T_acc': sum(not c['predicted_empty'] for c in positive) / len(positive)}


def run(args):
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output, source = Path(args.output), Path(args.source_cache)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / 'INDEX.json').read_text())
    parent_config = json.loads((Path(index['parent']) / 'CONFIG.json').read_text())
    if parent_config['old_faithful_inherited'] or parent_config['generated_data']:
        raise ValueError('retired/generated parent forbidden')
    rows = [json.loads(p.read_text()) for p in sorted((source / 'records').glob('*.json'))]
    training, heldout = select_training(rows, args.train_limit), select_holdout(rows, args.holdout_limit)
    if {r['image_id'] for r in training} & {r['image_id'] for r in heldout}:
        raise ValueError('training/held-out image overlap')
    selected = training + heldout
    config = {'adapter_version': 'evidence_routing_v2_calibrated_prior_balanced_alignment',
              'parent': index['parent'], 'parent_model_identity': index['parent_model_identity'],
              'source_index_sha256': digest_file(source / 'INDEX.json'),
              'training_ids': [r['id'] for r in training], 'heldout_ids': [r['id'] for r in heldout],
              'language_dim': parent_config['language_dim'], 'language_frozen': True, 'sam_frozen': True,
              'generated_data': False, 'old_faithful_inherited': False,
              'hard_negative_sampling': 'official TRAIN no-target examples with native proposals first',
              'existence_prior': 'learnable positive weight of native binary hint, initially 0.1',
              'alignment_loss': 'class-balanced BCE plus Dice, original masks only',
              'novelty_validated': False, 'public_evaluation_started': False}
    atomic_json(output / 'CONFIG.json', config)
    state = {'status': 'PREPARING_INPUT_FEATURES', 'pid': os.getpid(),
             'training_records': len(training), 'heldout_records': len(heldout),
             'language_frozen': True, 'sam_frozen': True, 'cases_done': 0, 'training_performed': True}

    def update(**values):
        state.update(values, updated_at_unix=time.time(), peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        atomic_json(output / 'STATUS.json', state)
        print(json.dumps(state), flush=True)

    update()
    reuse = Path(args.reuse_inputs) if args.reuse_inputs else None
    if reuse:
        old_config = json.loads((reuse / 'CONFIG.json').read_text())
        if any(old_config[key] != config[key] for key in ('parent', 'parent_model_identity', 'source_index_sha256')):
            raise ValueError('incompatible input feature reuse')
    runtime = None
    inputs_by_id = {}
    for number, record in enumerate(selected, 1):
        reusable = reuse / 'input_features' / (record['id'] + '.pth') if reuse else None
        if reusable and reusable.exists():
            inputs_by_id[record['id']] = torch.load(reusable, map_location='cpu', weights_only=True)
        else:
            proposals = torch.load(record['feature_path'], map_location='cpu', weights_only=True)
            if proposals:
                semantic = proposals[0]['semantic']
            else:
                if runtime is None:
                    runtime = FreshRuntime(args.model_dir, args.assets, index['parent'])
                    runtime.model.requires_grad_(False).eval()
                with Image.open(record['image']) as image:
                    semantic = runtime.prefix_feature(image.convert('RGB'), record['query'])
            inputs_by_id[record['id']] = pack_inputs(proposals, semantic, parent_config['empty_threshold'])
        update(status='PREPARING_INPUT_FEATURES', input_features_done=number, input_features_total=len(selected))
    if runtime is None:
        from projects.evoseg.dialog.runtime import NativeRuntime
        codec = NativeRuntime(args.model_dir, args.assets, load_language=False).tokenizer
    else:
        codec = runtime.tokenizer
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    input_dir = output / 'input_features'
    input_dir.mkdir(exist_ok=True)
    for key, inputs in inputs_by_id.items():
        torch.save(inputs, input_dir / (key + '.pth'))
    heads = {mode: EvidenceRoutingAdapter(config['language_dim']).cuda() for mode in ('global', 'spatial')}
    heads['global'].load_state_dict(heads['spatial'].state_dict(), strict=True)
    optimizers = {mode: torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=.01)
                  for mode, head in heads.items()}

    def evaluate(step):
        baseline, methods = {}, {mode: {} for mode in heads}
        for head in heads.values():
            head.eval()
        for number, record in enumerate(heldout, 1):
            inputs = to_device(inputs_by_id[record['id']])
            states = torch.load(record['sam_states'], map_location='cuda:0', weights_only=True)
            shape = read_mask(record['parent_prediction']).shape
            # All inference first; human masks are read only for scoring later.
            predictions = {mode: cached_prediction(head, codec, states, inputs, shape, mode)[0]
                           for mode, head in heads.items()}
            parent_prediction = read_mask(record['parent_prediction'])
            gt = read_mask(record['human_gt_mask'])
            valid = read_mask(record['human_valid_mask']) if record.get('human_valid_mask') else None
            malformed = record['parent_inference']['malformed_codes']
            baseline[record['id']] = metric_values(gt, parent_prediction, malformed, valid)
            for mode, prediction in predictions.items():
                if step == 0 and not np.array_equal(prediction, parent_prediction):
                    raise RuntimeError('zero-initialized evidence interface changed the native parent mask')
                methods[mode][record['id']] = metric_values(gt, prediction, malformed, valid)
            update(status='PRIVATE_HELDOUT_CHECK', training_step=step, cases_done=number,
                   expected_cases=len(heldout))
        image_ids = {r['id']: r['image_id'] for r in heldout}
        guards = {mode: preservation_guard(baseline, values, image_ids) for mode, values in methods.items()}
        report = {'step': step, 'reference': score_cases(baseline),
                  'scores': {mode: score_cases(values) for mode, values in methods.items()}, 'guards': guards,
                  'cases': methods, 'full_public_evaluation': False, 'sota_claimed': False}
        atomic_json(output / f'HELDOUT_{step:06d}.json', report)
        for head in heads.values():
            head.train()
        return report

    initial = evaluate(0)
    groups = {name: [r for r in training if r['bucket'] == name]
              for name in ('ref_positive', 'gref_positive', 'gref_empty', 'reason_positive')}
    history = []
    stopped_for_regression = False
    for step in range(1, args.steps + 1):
        bucket = tuple(groups)[(step - 1) % 4]
        record = random.Random(42 + step).choice(groups[bucket])
        inputs = to_device(inputs_by_id[record['id']])
        states = torch.load(record['sam_states'], map_location='cuda:0', weights_only=True)
        target, valid = targets(record)
        losses, gradients = {}, {}
        for mode, head in heads.items():
            optimizer = optimizers[mode]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = supervised_loss(head, codec, states, inputs, target, valid, mode, step)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite evidence routing loss')
            loss.backward()
            gradient = sum(float(p.grad.float().abs().sum()) for p in head.parameters() if p.grad is not None)
            if not np.isfinite(gradient) or gradient <= 0:
                raise RuntimeError('no evidence routing gradient')
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses[mode], gradients[mode] = parts, gradient
        update(status='MATCHED_EVIDENCE_TRAINING', step=step, planned_steps=args.steps,
               losses=losses, gradient_l1=gradients)
        if step in (25, 100, 300, args.steps):
            for mode, head in heads.items():
                destination = output / 'training' / mode / f'checkpoint_{step:06d}'
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(head.state_dict(), destination / 'EVIDENCE.pth')
                atomic_json(destination / 'CONFIG.json', {**config, 'mode': mode, 'step': step})
            if step in (100, 300) and step != args.steps:
                intermediate = evaluate(step)
                history.append({key: intermediate[key] for key in ('step', 'scores', 'guards')})
                if not all(guard['qualified'] for guard in intermediate['guards'].values()):
                    stopped_for_regression = True
                    break
    final = intermediate if stopped_for_regression else evaluate(step)
    # A small held-out pilot is not sufficient to trigger a public benchmark.
    atomic_json(output / 'REPORT.json', {'training_steps': step, 'requested_steps': args.steps,
        'stopped_for_regression': stopped_for_regression, 'history': history, 'initial_native_exact': True,
        'reference': final['reference'], 'scores': final['scores'], 'guards': final['guards'],
        'matched_global_spatial_training': True, 'training_performed': True,
        'language_frozen': True, 'sam_frozen': True, 'generated_data': False,
        'public_evaluation_started': False, 'sota_claimed': False})
    update(status='PRIVATE_EVIDENCE_PILOT_COMPLETE', cases_done=len(heldout))


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source-cache', default=base + '/results/evoseg_hallucination_20261007/region_public_pilot/cache')
    p.add_argument('--reuse-inputs')
    p.add_argument('--model-dir', default=base + '/models/Qwen3-VL-8B-SAMTok-official')
    p.add_argument('--assets', default=base + '/results/evoseg_dialog_20261006/ASSETS.json')
    p.add_argument('--train-limit', type=int, default=128)
    p.add_argument('--holdout-limit', type=int, default=64)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--gpu-memory-fraction', type=float, default=.4)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'ERROR.json', {'error': str(error), 'status': 'FAILED_CHECKPOINTS_PRESERVED'})
        raise


if __name__ == '__main__':
    main()

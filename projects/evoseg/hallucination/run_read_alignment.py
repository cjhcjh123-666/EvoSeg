"""Paired READ-style private interface diagnostic; never a public SOTA claim."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from projects.evoseg.dialog.runtime import NativeRuntime
from projects.evoseg.restart.prepare_foundation import atomic_json
from .eval_fresh import metric_values
from .protocol import read_mask
from .read_alignment import ALIGNMENT_VERSION, UPSTREAM_REVISION, aligned_predictions
from .train_fresh import cached_records


def select_records(rows, limit):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['id']):
        if row['holdout']:
            if row['source_split'] != 'train' or row['generated_query'] or row['pseudo_label']:
                raise ValueError('only held-out public TRAIN annotations are permitted')
            groups[row['bucket']].append(row)
    if not groups or limit < len(groups):
        raise ValueError('diagnostic budget must cover every source bucket')
    # Round-robin exact original expressions, not best/worst score selection.
    chosen = []
    for index in range(max(map(len, groups.values()))):
        for name in sorted(groups):
            if index < len(groups[name]):
                chosen.append(groups[name][index])
                if len(chosen) == limit:
                    return chosen
    return chosen


def aggregate(cases):
    report = {'cases': len(cases), 'quality_results_are_private_diagnostic_only': True,
              'public_benchmark': False, 'training_performed': False,
              'paper_score_reproduction': False, 'sota_claimed': False,
              'arms_share_one_generation': True, 'native_empty_cases': 0,
              'candidates_with_points': 0, 'per_bucket': {}}
    report['native_empty_cases'] = sum(c['inference']['native_empty'] for c in cases)
    report['candidates_with_points'] = sum(p.get('skip_reason') is None for c in cases
                                          for p in c['inference']['points'])
    for bucket in sorted({c['bucket'] for c in cases}):
        rows = [c for c in cases if c['bucket'] == bucket]
        report['per_bucket'][bucket] = {}
        for arm in ('native', 'read_style'):
            values = [c['metrics'][arm] for c in rows]
            report['per_bucket'][bucket][arm] = {
                'cases': len(values), 'mean_iou': sum(v['iou'] for v in values) / len(values),
                'cIoU': sum(v['intersection'] for v in values) / max(1, sum(v['union'] for v in values)),
                'predicted_empty': sum(v['predicted_empty'] for v in values)}
    return report


def run(args):
    if args.device == 'cuda':
        torch.cuda.set_device(0)
        if args.gpu_memory_fraction is not None:
            if not 0 < args.gpu_memory_fraction <= 1:
                raise ValueError('GPU memory fraction must be in (0, 1]')
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = select_records(cached_records(args.cache), args.limit)
    state = {'pid': os.getpid(), 'status': 'LOADING_FOUNDATION', 'device': args.device,
             'alignment_version': ALIGNMENT_VERSION, 'upstream_revision': UPSTREAM_REVISION,
             'expected_cases': len(records), 'cases_done': 0,
             'training_performed': False, 'paper_score_reproduction': False}

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'STATUS.json', state)
        print(json.dumps(state), flush=True)

    update()
    runtime = NativeRuntime(args.model_dir, args.assets, device=args.device)
    identity = {'foundation_revision': runtime.revision, 'weights': runtime.adapter_fingerprint,
                'alignment_version': ALIGNMENT_VERSION, 'upstream_revision': UPSTREAM_REVISION,
                'device': args.device, 'config': 'published_7b_thresholds_budget_geometry_adapted'}
    atomic_json(output / 'CONFIG.json', {**identity, 'record_ids': [r['id'] for r in records],
                'data': 'original queries from held-out public TRAIN images; no public test tuning',
                'label_input_to_inference': False, 'generated_data': False})
    cases = []
    for record in records:
        update(status='PAIRED_INFERENCE', current_case=record['id'])
        destination = output / 'cases' / (record['id'] + '.json')
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if destination.exists():
            case = json.loads(destination.read_text())
            if case['model_identity'] != identity or case['record_sha256'] != fingerprint:
                raise RuntimeError('stale alignment case cache')
        else:
            # No target path/label is read until ALL predictions are finalized.
            masks, audit = aligned_predictions(runtime, record['image'], record['query'])
            gt = read_mask(record['human_gt_mask'])
            valid = read_mask(record['human_valid_mask']) if record.get('human_valid_mask') else None
            metrics = {arm: metric_values(gt, mask, audit['malformed_codes'], valid)
                       for arm, mask in masks.items()}
            directory = output / 'masks' / record['id']
            directory.mkdir(parents=True, exist_ok=True)
            for arm, mask in masks.items():
                Image.fromarray(mask.astype(np.uint8) * 255).save(directory / (arm + '.png'))
            case = {'id': record['id'], 'query': record['query'], 'image': record['image'],
                    'bucket': record['bucket'], 'model_identity': identity,
                    'record_sha256': fingerprint, 'inference': audit, 'metrics': metrics}
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, case)
        cases.append(case)
        update(cases_done=len(cases), last_case=record['id'],
               peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None)
    atomic_json(output / 'REPORT.json', {**aggregate(cases), 'model_identity': identity,
                'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None})
    update(status='PRIVATE_ALIGNMENT_COMPLETE')


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', default=base + '/models/Qwen3-VL-8B-SAMTok-official')
    p.add_argument('--assets', default=base + '/results/evoseg_dialog_20261006/ASSETS.json')
    p.add_argument('--cache', default=base + '/results/evoseg_hallucination_20261007/reasoning_public_mix/cache')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--gpu-memory-fraction', type=float)
    p.add_argument('--limit', type=int, default=96)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'ERROR.json', {'error': str(error), 'pid': os.getpid(),
                                          'status': 'FAILED_PREDICTIONS_PRESERVED'})
        raise


if __name__ == '__main__':
    main()

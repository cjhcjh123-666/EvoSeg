"""Encode human GT as native training targets; no pseudo labels or rewritten queries."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from projects.evoseg.restart.prepare_foundation import atomic_json
from projects.evoseg.dialog.protocol import decode_annotation
from projects.evoseg.dialog.runtime import NativeRuntime, token_string
from .protocol import digest_file


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    source = json.loads(Path(args.manifest).read_text())
    manifest_digest = digest_file(args.manifest)
    if not source['source_only_train'] or source['generated_queries'] or source['pseudo_labels']:
        raise RuntimeError('public human-supervised source required')
    annotations = {name: {int(a['id']): a for a in json.loads(Path(path).read_text())['annotations']}
                   for name, path in source['annotation_sources'].items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = NativeRuntime(args.model_dir, args.assets, load_language=False)
    records = [r for r in source['records'] if r['image_id'] % world == rank]
    records.sort(key=lambda r: (r['image_id'], r['id']))
    state = {'status': 'CACHING_PUBLIC_GT', 'rank': rank, 'pid': os.getpid(),
             'records_done': 0, 'records_total': len(records), 'started_at_unix': time.time()}
    for record in records:
        destination = output / 'records' / (record['id'] + '.json')
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous['foundation_revision'] != runtime.revision or previous['source_manifest_sha256'] != manifest_digest:
                raise RuntimeError('cache identity changed')
        else:
            runtime.set_image(record['image'])
            image_features = output / 'sam_states' / (str(record['image_id']) + '.pth')
            image_features.parent.mkdir(parents=True, exist_ok=True)
            if not image_features.exists():
                temporary = image_features.with_suffix('.tmp')
                # Position tensors are unused by native mask encoding/decoding;
                # do not duplicate the large positional pyramid on disk.
                needed = {key: runtime.sam_states[key] for key in ('current_vision_feats', 'feat_sizes')}
                torch.save(cpu_tree(needed), temporary)
                temporary.replace(image_features)
            target = np.zeros((runtime.image.height, runtime.image.width), dtype=bool)
            if not record['private_no_target']:
                for annotation_id in record['private_annotation_ids']:
                    target |= decode_annotation(annotations[record['annotation_source']][annotation_id],
                                                *target.shape, record['image_id'])
                if not target.any():
                    raise RuntimeError('positive public annotation has no target pixels; do not relabel silently')
            gt_path = output / 'human_gt' / (record['id'] + '.png')
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target.astype(np.uint8) * 255).save(gt_path)
            codes = runtime.encode(target) if target.any() else None
            answer = '```json\n' + json.dumps([{'mask_2d': token_string(codes)}] if codes is not None else []) + '\n```'
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, {**record, 'sam_states': str(image_features), 'human_gt_mask': str(gt_path),
                'private_target_codes': codes, 'assistant_target': answer,
                'foundation_revision': runtime.revision, 'source_manifest_sha256': manifest_digest,
                'supervision': 'original human GT only; native code encoding is target-format conversion'})
        state.update(records_done=state['records_done'] + 1, updated_at_unix=time.time())
        if state['records_done'] % 16 == 0:
            atomic_json(output / f'STATUS_rank{rank}.json', state)
            print(json.dumps(state), flush=True)
    state.update(status='CACHE_SHARD_COMPLETE', finished_at_unix=time.time())
    atomic_json(output / f'STATUS_rank{rank}.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    p.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    run(p.parse_args())


if __name__ == '__main__':
    main()

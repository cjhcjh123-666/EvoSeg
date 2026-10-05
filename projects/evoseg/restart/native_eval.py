"""Durable native Uniform inference; shards never pad or duplicate expressions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from .native_model import load_release, restore_adaptation
from .prepare_foundation import atomic_json


def expression_tasks(metadata, keys=None):
    allowed = None if keys is None else {tuple(key) for key in keys}
    return [(video, expression) for video in sorted(metadata)
            for expression in sorted(metadata[video]['expressions'])
            if allowed is None or (video, expression) in allowed]


def native_query(expression):
    # The upstream v1 evaluator left '?' expressions unchanged and omitted
    # <image>. MeViS-v2 includes reasoning questions; retain their exact text
    # and supply the mandatory image placeholder in both baseline and FT.
    if '?' in expression:
        return expression if '<image>' in expression else '<image>\n' + expression
    return '<image>\nPlease segment {}.'.format(expression)


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.meta).read_text())['videos']
    keys = json.loads(Path(args.keys).read_text()) if args.keys else None
    tasks = expression_tasks(metadata, keys)[rank::world]
    model, tokenizer, loading = load_release(args.model_dir, args.assets, torch.device('cuda', local))
    if args.checkpoint:
        checkpoint = restore_adaptation(model, args.checkpoint)
        if checkpoint['foundation_revision'] != loading['revision']:
            raise RuntimeError('adaptation foundation revision differs from this evaluation')
    atomic_json(output / f'LOADING_rank{rank}.json', loading)
    state = {'rank': rank, 'world_size': world, 'tasks': len(tasks), 'mode': 'uniform',
             'checkpoint': args.checkpoint, 'completed': 0, 'status': 'EVALUATING',
             'started_at_unix': time.time(), 'dataset_meta': args.meta}
    status_path = output / f'STATUS_rank{rank}.json'
    atomic_json(status_path, state)
    cached_video, images = None, None
    for task_number, (video, expression_id) in enumerate(tasks):
        destination = output / 'cases' / video / f'{expression_id}.json'
        if destination.exists():
            # A job's output directory is tied to one immutable checkpoint.
            previous = json.loads(destination.read_text())
            if previous['checkpoint'] != args.checkpoint or previous['foundation_revision'] != loading['revision']:
                raise RuntimeError('existing predictions belong to another model')
        else:
            if cached_video != video:
                images = []
                for frame in sorted(metadata[video]['frames']):
                    with Image.open(Path(args.images) / video / (frame + '.jpg')) as image:
                        images.append(image.convert('RGB'))
                cached_video = video
            expression = metadata[video]['expressions'][expression_id]['exp']
            query = native_query(expression)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                result = model.predict_forward(video=images, text=query, tokenizer=tokenizer,
                                               mode='uniform', samurai=False)
            predictions = result['prediction_masks']
            if len(predictions) != 1 or predictions[0].shape != (len(images), images[0].height, images[0].width):
                raise RuntimeError('native inference did not return one mask for every original frame')
            rles = []
            for mask in predictions[0]:
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode()
                rles.append(rle)
            item = {'video_id': video, 'exp_id': expression_id, 'frames': sorted(metadata[video]['frames']),
                    'prediction_masks': rles, 'text_prediction': result['prediction'],
                    'checkpoint': args.checkpoint, 'foundation_revision': loading['revision']}
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, item)
        state.update(completed=task_number + 1, updated_at_unix=time.time(),
                     gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        atomic_json(status_path, state)
        print(json.dumps({**state, 'case': [video, expression_id]}), flush=True)
    state.update(status='SHARD_COMPLETE', finished_at_unix=time.time())
    atomic_json(status_path, state)


def merge_predictions(metadata_file, output_dir, keys=None):
    metadata = json.loads(Path(metadata_file).read_text())['videos']
    tasks = expression_tasks(metadata, keys)
    results = {}
    for video, expression in tasks:
        item = json.loads((Path(output_dir) / 'cases' / video / f'{expression}.json').read_text())
        if item['frames'] != sorted(metadata[video]['frames']) or len(item['prediction_masks']) != len(item['frames']):
            raise RuntimeError('prediction/annotation frame inventory differs')
        results.setdefault(video, {})[expression] = item
    atomic_json(Path(output_dir) / 'results.json', results)
    return len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'assets', 'images', 'meta', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--keys')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        atomic_json(Path(args.output) / f'FAILURE_rank{os.environ.get("RANK", "0")}.json',
                    {'error_type': type(error).__name__, 'error': str(error), 'success': False,
                     'updated_at_unix': time.time()})
        raise


if __name__ == '__main__':
    main()

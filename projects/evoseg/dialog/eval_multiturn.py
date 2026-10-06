"""Full public MRSeg inputs; GT and predicted history are distinct evaluations.

Native history is reconstructed from raw public markers, not claimed identical
to SAMTok's unreleased sampled/preprocessed benchmark files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mask_utils
import torch

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import (PROTOCOL_VERSION, RoundMetrics, assistant_history, decode_annotation, render_query)
from .public_data import parse_conversation
from .runtime import NativeRuntime, parse_mask_pair


DATASETS = ('refcoco', 'refcoco+', 'refcocog')


def save_case(destination, values):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, values)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = NativeRuntime(args.model_dir, args.assets)
    annotations = {int(item['id']): item for item in json.loads(Path(args.annotations).read_text())['annotations']}
    state = {'rank': rank, 'world_size': world, 'status': 'EVALUATING', 'history_mode': args.history_mode,
             'protocol': PROTOCOL_VERSION, 'started_at_unix': time.time(), 'completed_dialogues': 0,
             'completed_rounds': 0, 'training_started': False}
    atomic_json(output / f'LOADING_rank{rank}.json', {'language_loading_info': runtime.loading,
                'specialized_mask_tokenizer_strict': True, 'foundation_revision': runtime.revision})
    for dataset in DATASETS:
        path = Path(args.dialogue_root) / f'mr_{dataset}_val.json'
        records = json.loads(path.read_text())
        selected = records if args.limit is None else records[:args.limit]
        digest = file_digest(path)
        for index in range(rank, len(selected), world):
            destination = output / dataset / 'cases' / f'{index:06d}.json'
            turns = parse_conversation(selected[index])
            if destination.exists():
                previous = json.loads(destination.read_text())
                if (previous['history_mode'] != args.history_mode or previous['data_sha256'] != digest or
                    previous['foundation_revision'] != runtime.revision or previous['protocol'] != PROTOCOL_VERSION):
                    raise RuntimeError('cached dialogue predictions use another model or protocol')
                if len(previous['rounds']) != len(turns):
                    raise RuntimeError('cached dialogue has incomplete round coverage')
            else:
                image_path = Path(args.coco_images) / Path(turns[0]['source_image']).name
                runtime.set_image(image_path)
                image_id = int(image_path.stem.split('_')[-1])
                targets, gt_codes = {}, {}
                def target_mask(annotation_id):
                    if annotation_id not in targets:
                        targets[annotation_id] = decode_annotation(annotations[annotation_id],
                            runtime.image.height, runtime.image.width, image_id)
                    return targets[annotation_id]
                def ground_truth_codes(annotation_id):
                    if annotation_id not in gt_codes:
                        gt_codes[annotation_id] = runtime.encode(target_mask(annotation_id))
                    return gt_codes[annotation_id]
                messages, history, results = [], [], []
                for turn in turns:
                    query = render_query(turn, history, args.history_mode, ground_truth_codes)
                    content = [{'type': 'text', 'text': query}]
                    if turn['round'] == 1:
                        content.insert(0, {'type': 'image', 'image': runtime.image})
                    messages.append({'role': 'user', 'content': content})
                    answer = runtime.generate(messages)
                    codes = parse_mask_pair(answer)
                    prediction = runtime.decode(codes)
                    target = target_mask(turn['source_annotation_id'])
                    metrics = RoundMetrics()
                    metrics.add(turn['round'], target, prediction, invalid=codes is None)
                    rle = mask_utils.encode(np.asfortranarray(prediction.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode()
                    results.append({'round': turn['round'], 'native_answer': answer, 'codes': codes,
                                    'prediction_rle': rle, **metrics.summary()[str(turn['round'])]})
                    previous_answer, previous_codes = assistant_history(args.history_mode, answer, codes,
                        lambda: ground_truth_codes(turn['source_annotation_id']))
                    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': previous_answer}]})
                    history.append({'codes': previous_codes})
                save_case(destination, {'dataset': dataset, 'index': index, 'image': str(image_path),
                    'foundation_revision': runtime.revision, 'protocol': PROTOCOL_VERSION,
                    'data_sha256': digest, 'history_mode': args.history_mode, 'rounds': results,
                    'gt_mask_encodings_for_history': len(gt_codes)})
            state.update(completed_dialogues=state['completed_dialogues'] + 1,
                         completed_rounds=state['completed_rounds'] + len(turns),
                         dataset=dataset, case=index, updated_at_unix=time.time(),
                         gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            atomic_json(output / f'STATUS_rank{rank}.json', state)
            print(json.dumps(state), flush=True)
    state.update(status='SHARD_COMPLETE', finished_at_unix=time.time())
    atomic_json(output / f'STATUS_rank{rank}.json', state)


def summarize(args):
    output = Path(args.output)
    report = {'history_mode': args.history_mode, 'protocol': PROTOCOL_VERSION,
              'official_raw_files': True, 'paper_sampled_preprocessed_subset': False,
              'source_report_comparability': 'pending exact original preprocessing/subset confirmation',
              'complete_public_evaluation': args.limit is None, 'datasets': {},
              'training_started': False, 'sota_claimed': False}
    for dataset in DATASETS:
        path = Path(args.dialogue_root) / f'mr_{dataset}_val.json'
        records = json.loads(path.read_text())
        selected = records if args.limit is None else records[:args.limit]
        accumulator = RoundMetrics()
        for index, record in enumerate(selected):
            item = json.loads((output / dataset / 'cases' / f'{index:06d}.json').read_text())
            expected = parse_conversation(record)
            if len(item['rounds']) != len(expected) or item['history_mode'] != args.history_mode:
                raise RuntimeError('evaluation round coverage or history policy differs')
            for values in item['rounds']:
                target = accumulator.rounds[int(values['round'])]
                for key in target:
                    target[key] += values[key]
        report['datasets'][dataset] = {'dialogues': len(selected), 'rounds': accumulator.summary()}
    average = {}
    for round_id in sorted({key for dataset in report['datasets'].values() for key in dataset['rounds']}, key=int):
        available = [dataset['rounds'][round_id]['cIoU'] for dataset in report['datasets'].values()
                     if round_id in dataset['rounds']]
        average[round_id] = {'dataset_count': len(available), 'mean_dataset_cIoU': sum(available) / len(available)}
    report['mean_across_datasets_by_round'] = average
    atomic_json(output / 'METRICS.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    parser.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    parser.add_argument('--dialogue-root', default=str(base / 'datasets/SegLLM-official/conversations_folder/all_data_mix_val'))
    parser.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    parser.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    parser.add_argument('--history-mode', choices=['gt_history', 'predicted_history'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, help='Explicit diagnostic only, never a full benchmark')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('diagnostic limit must be positive')
    if args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

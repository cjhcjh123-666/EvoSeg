"""Public-only mini training cache; no validation images or generated drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import assistant_history, decode_annotation, render_query
from .public_data import parse_conversation
from .runtime import NativeRuntime


def build(args):
    root = Path(args.dialogue_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'MANIFEST.json').exists():
        raise RuntimeError('training cache already exists; inspect before replacing')
    validation_images = set()
    for path in (root / 'all_data_mix_val').glob('mr_ref*_val.json'):
        for record in json.loads(path.read_text()):
            validation_images.add(parse_conversation(record)[0]['source_image'])
    annotations = {int(item['id']): item for item in json.loads(Path(args.annotations).read_text())['annotations']}
    runtime = NativeRuntime(args.model_dir, args.assets, load_language=False)
    manifest = {'status': 'PREPARING_PUBLIC_TRAIN_CACHE', 'started_at_unix': time.time(),
                'training_started': False, 'validation_images_excluded': True,
                'foundation_revision': runtime.revision, 'generated_drafts_included': False,
                'completed_dialogues': 0, 'completed_rounds': 0, 'per_dataset': {},
                'history_mode': 'GT_training_history', 'cache_scope': 'bounded public pilot, not full training corpus'}
    for dataset in ('refcoco', 'refcoco+', 'refcocog'):
        source = root / 'all_data_mix_train' / f'mr_{dataset}_train.json'
        records = json.loads(source.read_text())
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        selected = []
        for index, record in enumerate(records):
            turns = parse_conversation(record)
            if turns[0]['source_image'] not in validation_images:
                selected.append((index, turns))
            if len(selected) == args.dialogues_per_dataset:
                break
        if not selected:
            raise RuntimeError('no disjoint public training dialogues: ' + dataset)
        manifest['per_dataset'][dataset] = {'requested_dialogues': args.dialogues_per_dataset,
                                             'selected_dialogues': len(selected), 'source_sha256': source_sha}
        for index, turns in selected:
            image_name = turns[0]['source_image']
            image_path = Path(args.coco_images) / Path(image_name).name
            runtime.set_image(image_path)
            image_id = int(image_path.stem.split('_')[-1])
            feature_path = output / 'image_features' / f'{image_id}.pt'
            if not feature_path.exists():
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                states = runtime.sam_states
                size = states['feat_sizes'][-1]
                features = states['current_vision_feats'][-1].permute(1, 2, 0).reshape(1, 256, *size)[0]
                temporary = feature_path.with_suffix('.tmp')
                torch.save(features.detach().cpu().contiguous(), temporary)
                temporary.replace(feature_path)
            codes, targets = {}, {}
            def gt_codes(anno_id):
                if anno_id not in codes:
                    targets[anno_id] = decode_annotation(annotations[anno_id], runtime.image.height,
                                                         runtime.image.width, image_id)
                    codes[anno_id] = runtime.encode(targets[anno_id])
                return codes[anno_id]
            history, compiled = [], []
            for turn in turns:
                query = render_query(turn, history, 'gt_history', gt_codes)
                current_codes = gt_codes(turn['source_annotation_id'])
                answer, _ = assistant_history('gt_history', '', None, lambda: current_codes)
                compiled.append({'round': turn['round'], 'query': query, 'assistant_target': answer,
                                 'private_supervision': {'witness_rounds': turn['reference_rounds'],
                                     'edit_rounds': [], 'operation': 'new',
                                     'target_annotation_id': turn['source_annotation_id'], 'target_codes': current_codes}})
                history.append({'codes': current_codes})
            case = output / dataset / f'{index:06d}.json'
            case.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(case, {'dataset': dataset, 'source_index': index, 'source_sha256': source_sha,
                              'image': str(image_path), 'image_feature_cache': str(feature_path),
                              'foundation_revision': runtime.revision, 'compiled_turns': compiled,
                              'supervision_source': 'public_MRSeg_and_original_COCO_GT_only',
                              'editing_branch_supervised': False})
            manifest['completed_dialogues'] += 1
            manifest['completed_rounds'] += len(turns)
            atomic_json(output / 'STATUS.json', manifest)
            if manifest['completed_dialogues'] % 32 == 0:
                print(json.dumps(manifest), flush=True)
    manifest.update(status='PUBLIC_TRAIN_PILOT_CACHE_READY', finished_at_unix=time.time(),
                    editing_branch_supervised=False,
                    next_gate='verify native LoRA gradients and integrate witness/scope controls; no training has run')
    atomic_json(output / 'MANIFEST.json', manifest)
    atomic_json(output / 'STATUS.json', manifest)
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--dialogue-root', default=str(base / 'datasets/SegLLM-official/conversations_folder'))
    parser.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    parser.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    parser.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    parser.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    parser.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/public_train_pilot_cache'))
    parser.add_argument('--dialogues-per-dataset', type=int, default=256)
    args = parser.parse_args()
    if args.dialogues_per_dataset < 1:
        parser.error('dialogue count must be positive')
    build(args)


if __name__ == '__main__':
    main()

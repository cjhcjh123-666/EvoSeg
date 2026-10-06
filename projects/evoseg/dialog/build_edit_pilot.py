"""Small generated-language pilot with public video mask supervision only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from pycocotools import mask as mask_utils
from PIL import Image

from projects.evoseg.restart.native_data import split_video_ids
from projects.evoseg.restart.prepare_foundation import atomic_json
from .state import SelectionLedger, mask_recipe


def make_turns(first_expression, second_expression, registry):
    ledger = SelectionLedger(frozenset(registry))
    instructions = [
        ('select', ['A'], f'Segment the target described as: "{first_expression}".', []),
        ('add', ['B'], f'Also add the target described as: "{second_expression}". Keep the first selection.', [1]),
        ('remove', ['A'], 'Remove the first target from the current selection; keep the one added later.', [1, 2]),
        ('undo', [], 'Undo that removal and restore the previous selection.', [3]),
        ('select', ['A'], 'Keep only the target from my first request.', [1]),
    ]
    turns = []
    for round_id, (operation, targets, text, reference_turns) in enumerate(instructions, 1):
        before = sorted(ledger.active)
        active = ledger.apply(operation, targets)
        turns.append({'round': round_id, 'instruction': text,
                      'supervision': {'operation': operation, 'resolved_targets': targets,
                                      'reference_turns': reference_turns,
                                      'active_before': before, 'active_after': sorted(active),
                                      'mask_recipe': {'operation': 'union_public_annotation_tracks',
                                                      'anno_ids': mask_recipe(active, registry)}}})
    return turns


def source_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(args):
    source = Path(args.source_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'train_edit_pilot.jsonl'
    if destination.exists():
        raise RuntimeError('pilot already exists; do not overwrite annotations')
    metadata_path, masks_path = source / 'meta_expressions_v2.json', source / 'mask_dict.json'
    videos = json.loads(metadata_path.read_text())['videos']
    masks = json.loads(masks_path.read_text())
    train_ids, heldout_ids = split_video_ids(videos, args.seed)
    order = sorted(train_ids, key=lambda key: hashlib.sha256(f'edit-pilot:{args.seed}:{key}'.encode()).hexdigest())
    examples, skipped = [], {}
    for video in order:
        if len(examples) >= args.count:
            break
        info = videos[video]
        expressions = info['expressions']
        candidates = [key for key in sorted(expressions) if expressions[key]['anno_id']]
        random.Random(f'edit-pilot:{args.seed}:{video}').shuffle(candidates)
        pair = next(((a, b) for a in candidates for b in candidates if a != b and
                     set(expressions[a]['anno_id']).isdisjoint(expressions[b]['anno_id'])), None)
        if pair is None:
            skipped['no_disjoint_public_targets'] = skipped.get('no_disjoint_public_targets', 0) + 1
            continue
        frames = info['frames']
        image_dir = source / 'JPEGImages' / video
        available = {path.name for path in image_dir.iterdir()}
        if not frames or any(frame + '.jpg' not in available for frame in frames):
            raise RuntimeError('missing public source frame: ' + video)
        with Image.open(image_dir / (frames[0] + '.jpg')) as image:
            shape = (image.height, image.width)
        registry = {alias: {'expression_id': key, 'expression': expressions[key]['exp'],
                            'anno_ids': expressions[key]['anno_id']}
                    for alias, key in zip(('A', 'B'), pair)}
        visible = []
        for target in registry.values():
            present = [False] * len(frames)
            for anno in target['anno_ids']:
                annotations = masks[str(anno)]
                if len(annotations) != len(frames):
                    raise RuntimeError('mask track/frame length mismatch: ' + video)
                for frame, encoded in enumerate(annotations):
                    if encoded is not None:
                        if tuple(encoded['size']) != shape:
                            raise RuntimeError('public mask/image dimensions mismatch: ' + video)
                        present[frame] |= int(mask_utils.area(encoded)) > 0
            visible.append(present)
        if not any(a and b for a, b in zip(*visible)):
            skipped['no_covisible_frame'] = skipped.get('no_covisible_frame', 0) + 1
            continue
        turns = make_turns(registry['A']['expression'], registry['B']['expression'], registry)
        examples.append({'id': 'edit-pilot-' + video, 'source_dataset': 'MeViS-v2',
                         'source_split': 'train', 'source_video_id': video,
                         'frames': frames, 'image_directory': str(image_dir),
                         'private_supervision_registry': registry, 'turns': turns,
                         'provenance': {'language': 'deterministic_templates_plus_verbatim_public_expressions',
                                        'masks': 'existing_public_GT_tracks_and_exact_set_composition',
                                        'human_review': 'PENDING', 'is_official_benchmark': False},
                         'history_policy': 'use_model_predictions_in_closed_loop; never expose private registry at inference'})
        if len(examples) % 50 == 0:
            print(f'validated draft dialogues: {len(examples)}/{args.count}', flush=True)
    if not examples:
        raise RuntimeError('no valid public video dialogue examples')
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + '\n')
    temporary.replace(destination)
    review = sorted(example['id'] for example in examples)[:min(100, len(examples))]
    atomic_json(output / 'REVIEW_QUEUE.json', {'example_ids': review, 'review_completed': False})
    atomic_json(output / 'MANIFEST.json', {'status': 'DRAFTS_GENERATED_REVIEW_REQUIRED',
               'requested_dialogues': args.count, 'generated_dialogues': len(examples),
               'turns': sum(len(example['turns']) for example in examples),
               'source_metadata_sha256': source_hash(metadata_path),
               'source_masks_sha256': source_hash(masks_path), 'seed': args.seed,
               'excluded_finetune_heldout_videos': heldout_ids, 'skipped': skipped,
               'training_started': False, 'paid_model_calls': 0, 'human_validated': False,
               'use': 'draft training-data pipeline pilot only; not test benchmark or human dialogues'})
    print(json.dumps({'generated_dialogues': len(examples), 'destination': str(destination),
                      'human_review': 'PENDING', 'training_started': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--source-root', default=str(base / 'datasets/mevis_v2/train'))
    parser.add_argument('--output-dir', default=str(base / 'datasets/EvoSeg-Dialog-Pilot-v0'))
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('dialogue count must be positive')
    build(args)


if __name__ == '__main__':
    main()

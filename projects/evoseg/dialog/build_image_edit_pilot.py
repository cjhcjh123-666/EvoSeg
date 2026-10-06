"""Image edit drafts from public MRSeg train; all public val images excluded."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from projects.evoseg.restart.prepare_foundation import atomic_json
from .build_edit_pilot import make_turns
from .protocol import decode_annotation
from .public_data import IMAGE, parse_conversation


def build(args):
    root = Path(args.dialogue_root)
    annotations = {int(item['id']): item for item in json.loads(Path(args.annotations).read_text())['annotations']}
    excluded = set()
    for path in (root / 'all_data_mix_val').glob('mr_ref*_val.json'):
        for record in json.loads(path.read_text()):
            excluded.add(parse_conversation(record)[0]['source_image'])
    groups = {}
    for name in ('refcoco', 'refcoco+', 'refcocog'):
        path = root / 'all_data_mix_train' / f'mr_{name}_train.json'
        for index, record in enumerate(json.loads(path.read_text())):
            first = parse_conversation(record)[0]
            if first['source_image'] in excluded or first['external_visual_references']:
                continue
            item = {'anno_ids': [first['source_annotation_id']], 'query': IMAGE.sub('', first['query']).strip(),
                    'source_file': path.name, 'source_dialogue_index': index}
            groups.setdefault(first['source_image'], {}).setdefault(first['source_annotation_id'], item)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file = destination / 'train_image_edit_pilot.jsonl'
    if file.exists():
        raise RuntimeError('image edit drafts already exist; do not overwrite them')
    examples = []
    order = sorted(groups, key=lambda image: hashlib.sha256(f'image-edits:{args.seed}:{image}'.encode()).hexdigest())
    for image_name in order:
        pool = groups[image_name]
        if len(pool) < 2:
            continue
        image_path = Path(args.coco_images) / Path(image_name).name
        with Image.open(image_path) as image:
            height, width = image.height, image.width
        image_id = int(image_path.stem.split('_')[-1])
        ids = sorted(pool)[:2]
        if any(not decode_annotation(annotations[anno], height, width, image_id).any() for anno in ids):
            continue
        registry = {alias: pool[anno] for alias, anno in zip(('A', 'B'), ids)}
        # Original first-turn requests are preserved verbatim, not turned into
        # invented object attributes or teacher-generated descriptions.
        turns = make_turns(registry['A']['query'], registry['B']['query'], registry)
        turns[0]['instruction'] = registry['A']['query']
        turns[1]['instruction'] = 'Add the result of this request to the current selection: ' + registry['B']['query']
        for turn in turns:
            operation = turn['supervision']['operation']
            turn['supervision']['router_operation'] = ('new' if turn['round'] == 1 else 'replace') if operation == 'select' else operation
        examples.append({'id': f'image-edit-{image_id}', 'image': str(image_path),
                         'source_image': image_name, 'source_split': 'public_MRSeg_train',
                         'private_supervision_registry': registry, 'turns': turns,
                         'provenance': {'language': 'templates_plus_verbatim_public_first_turn_requests',
                                        'masks': 'original_COCO_instance_GT_exact_object_set_unions',
                                        'human_review': 'PENDING', 'is_official_benchmark': False}})
        if len(examples) >= args.count:
            break
    if not examples:
        raise RuntimeError('no valid image-edit drafts')
    temporary = file.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + '\n')
    temporary.replace(file)
    atomic_json(destination / 'IMAGE_EDIT_MANIFEST.json', {
        'status': 'DRAFTS_GENERATED_REVIEW_REQUIRED', 'generated_dialogues': len(examples),
        'turns': sum(len(example['turns']) for example in examples), 'all_public_val_images_excluded': True,
        'public_val_image_count': len(excluded), 'human_validated': False, 'training_started': False,
        'available_operations': ['new', 'add', 'remove', 'undo', 'replace'],
        'refine_supervision_available': False})
    atomic_json(destination / 'IMAGE_REVIEW_QUEUE.json', {'example_ids': [example['id'] for example in examples[:100]],
                                                         'review_completed': False})
    print(json.dumps({'generated_image_dialogues': len(examples), 'public_val_overlap': 0,
                      'human_review': 'PENDING', 'training_started': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--dialogue-root', default=str(base / 'datasets/SegLLM-official/conversations_folder'))
    parser.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    parser.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    parser.add_argument('--output-dir', default=str(base / 'datasets/EvoSeg-Dialog-Pilot-v0'))
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('draft count must be positive')
    build(args)


if __name__ == '__main__':
    main()

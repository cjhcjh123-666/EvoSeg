"""Read public MRSeg markers without inventing labels or hiding protocol details."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from projects.evoseg.restart.prepare_foundation import atomic_json


IMAGE = re.compile(r'\[IMAGE\d+:([^\]]+)\]')
OUTPUT = re.compile(r'\[MASK-DECODE:([^\]]+)\]')
REFERENCE = re.compile(r'\[(?:MASK|BOX)-ENCODE:([^\]]+)\]')


def parse_conversation(record):
    turns, previous, image = [], {}, None
    conversations = record['conversations']
    if len(conversations) % 2:
        raise ValueError('public dialogue has an unmatched turn')
    for offset in range(0, len(conversations), 2):
        query, answer = conversations[offset:offset + 2]
        if query['from'] != 'human' or answer['from'] != 'gpt':
            raise ValueError('public dialogue roles do not alternate')
        images = IMAGE.findall(query['value'])
        if images:
            if len(set(images)) != 1 or image not in (None, images[0]):
                raise ValueError('multi-image dialogue requires its own explicit protocol')
            image = images[0]
        decoded = OUTPUT.findall(answer['value'])
        if image is None or len(decoded) != 1:
            raise ValueError('need one source image and one public output annotation per turn')
        payload = decoded[0].split(':')
        if len(payload) != 5 or not payload[2].isdigit():
            raise ValueError('unsupported public output marker; never guess annotation identifiers')
        relation, reference, target, target_image, dataset = payload
        if target_image != image:
            raise ValueError('public target mask refers to another image')
        references = []
        external = []
        for encoded in REFERENCE.findall(query['value']):
            fields = encoded.split('|')
            if len(fields) != 3 or fields[0] != image or not fields[2].isdigit():
                raise ValueError('unsupported public history marker')
            anno = int(fields[2])
            if anno not in previous:
                if offset == 0:
                    # Public *_hard_train contains a user-provided visual
                    # reference on turn one, not a nonexistent prior answer.
                    external.append(anno)
                    continue
                raise ValueError('history refers to an unobserved public mask')
            references.extend(previous[anno])
        round_id = offset // 2 + 1
        pointer = query.get('ind')
        if pointer is not None:
            if not isinstance(pointer, int) or not 1 <= pointer < round_id:
                raise ValueError('public history pointer is out of range')
            references = [pointer]
        turns.append({'round': round_id, 'query': query['value'], 'source_dataset': dataset,
                      'source_annotation_id': int(target), 'source_image': image,
                      'relation_type': relation, 'reference_rounds': sorted(set(references)),
                      'external_visual_references': sorted(set(external)),
                      'history_markers_require_protocol': bool(references or external)})
        previous.setdefault(int(target), []).append(round_id)
    return turns


def audit(data_dir, coco_images, annotations, output):
    annotation_data = json.loads(Path(annotations).read_text())
    ann_index = {int(item['id']): item for item in annotation_data['annotations']}
    summary = {'status': 'SCHEMA_AUDIT', 'files': {},
               'history_protocols': ['official_GT_history_if_required', 'separate_predicted_history_closed_loop'],
               'training_started': False}
    for path in sorted(Path(data_dir).glob('conversations_folder/*/mr_*.json')):
        records = json.loads(path.read_text())
        counter = Counter()
        errors = []
        for record in records:
            try:
                turns = parse_conversation(record)
                counter['parsed_dialogues'] += 1
                counter['rounds'] += len(turns)
                counter['history_dependent_rounds'] += sum(bool(turn['reference_rounds']) for turn in turns)
                counter['external_visual_reference_rounds'] += sum(bool(turn['external_visual_references']) for turn in turns)
                image = Path(coco_images) / Path(turns[0]['source_image']).name
                counter['local_images_available'] += int(image.is_file())
                for turn in turns:
                    if turn['source_dataset'] in ('refcoco', 'refcoco+', 'refcocog'):
                        counter['referring_masks'] += 1
                        ann = ann_index.get(turn['source_annotation_id'])
                        counter['local_referring_masks_available'] += int(ann is not None)
                        if ann is not None:
                            image_id = int(Path(turn['source_image']).stem.split('_')[-1])
                            if int(ann['image_id']) != image_id:
                                raise ValueError('annotation ID is present but belongs to another image')
                    else:
                        counter['non_refcoco_masks_need_source_audit'] += 1
            except (KeyError, ValueError) as error:
                counter['schema_failures'] += 1
                if len(errors) < 3:
                    errors.append(str(error))
        summary['files'][str(path.relative_to(data_dir))] = {
            'records': len(records), **counter, 'example_schema_failures': errors}
        print(path.name, dict(counter), flush=True)
    atomic_json(Path(output), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--data-dir', default=str(base / 'datasets/SegLLM-official'))
    parser.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    parser.add_argument('--annotations', default=str(base / 'datasets/grefcoco/instances.json'))
    parser.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/PUBLIC_DATA_AUDIT.json'))
    args = parser.parse_args()
    audit(args.data_dir, args.coco_images, args.annotations, args.output)


if __name__ == '__main__':
    main()

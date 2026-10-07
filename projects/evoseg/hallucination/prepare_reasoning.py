"""Audit original published ReasonSeg train annotations and cache a public mix.

No explanatory generations, new queries, edited images, or pseudo-labels.
The author's release contains paraphrases; only its existing first query is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import requests

from projects.evoseg.restart.prepare_foundation import atomic_json
from .public_mix import parquet_rows
from .protocol import digest_file, public_pairs

ANNOTATION_REPO = 'fcxfcx/ReasonSeg'
ANNOTATION_REVISION = 'f994bd3fdd9254dc76c33d9ed7784506be4ec639'
PARQUET_REPO = 'Zilun/ReasonSeg_train_withbbox'
PARQUET_REVISION = '622ca67e5f2cfc69400f4d60909535f02301fa6a'
PARQUET_HASHES = {
    'train-00000-of-00002.parquet': 'e2fd2f58b40ed59b0c977a7583548eff4714628d868cb0765edb8f2d518fb60b',
    'train-00001-of-00002.parquet': '75f78df489502c8f65c52316fa984da99d28a5d07b6f44f90ef9aa743634c1ef',
}


def annotation_masks(annotation, shape):
    """Match published LISA rasterization: decreasing area, smaller polys last."""
    polygons = []
    for item in annotation['shapes']:
        if item['label'].lower() == 'flag':
            continue
        points = np.array([item['points']], dtype=np.int32)
        temporary = np.zeros(shape, dtype=np.uint8)
        cv2.polylines(temporary, points, True, 1, 1)
        cv2.fillPoly(temporary, points, 1)
        polygons.append((int(temporary.sum()), item))
    order = np.argsort([area for area, _ in polygons])[::-1]
    mask = np.zeros(shape, dtype=np.uint8)
    for index in order:
        item = polygons[int(index)][1]
        value = 255 if 'ignore' in item['label'].lower() else 1
        points = np.array([item['points']], dtype=np.int32)
        cv2.polylines(mask, points, True, value, 1)
        cv2.fillPoly(mask, points, value)
    return mask == 1, mask != 255


def session(proxy):
    value = requests.Session()
    value.trust_env = False
    value.proxies = {'http': proxy, 'https': proxy}
    return value


def prepare(args):
    root, data = Path(args.output), Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    api = session(args.proxy)
    info = api.get(f'https://huggingface.co/api/datasets/{ANNOTATION_REPO}/revision/{ANNOTATION_REVISION}',
                   params={'blobs': 'true'}, timeout=30)
    info.raise_for_status()
    remote = {p['rfilename']: p for p in info.json()['siblings']}
    names = sorted(name for name in remote if name.startswith('train/') and name.endswith('.json'))
    if len(names) != 239:
        raise RuntimeError('unexpected published ReasonSeg training annotation coverage')

    def fetch(name):
        destination = root / 'original_annotations' / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            response = session(args.proxy).get(f'https://huggingface.co/datasets/{ANNOTATION_REPO}/resolve/{ANNOTATION_REVISION}/{name}', timeout=30)
            response.raise_for_status()
            temporary = destination.with_suffix('.tmp')
            temporary.write_bytes(response.content)
            temporary.replace(destination)
        raw = destination.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != remote[name]['blobId']:
            raise RuntimeError('published annotation Git-blob mismatch: ' + name)
        return Path(name).stem, json.loads(raw)

    with ThreadPoolExecutor(max_workers=6) as workers:
        annotations = dict(workers.map(fetch, names))
    forbidden = set()
    for folder in ('reasonseg_val', 'reasonseg_test'):
        for path in (data / folder).glob('*.parquet'):
            forbidden.update(str(r['image_id']) for r in parquet_rows(path, ['image_id']))
    for pair in public_pairs(data / 'hallusegbench/test'):
        forbidden.add(Path(pair['factual_image']).stem)
    original = json.loads((Path(args.source_run) / 'TRAIN_MANIFEST.json').read_text())
    records, mirror_files, seen = [], [], set()
    for filename, expected in PARQUET_HASHES.items():
        path = data / 'reasonseg_train' / filename
        actual = digest_file(path)
        if actual != expected:
            raise RuntimeError('ReasonSeg mirror parquet checksum mismatch')
        mirror_files.append({'filename': filename, 'sha256': actual})
        # Full-resolution mask arrays are large; never turn a whole shard into
        # Python nested lists at once (the generic metadata reader batches 1024).
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            name = str(row['image_id'])
            if name in seen or name in forbidden or name not in annotations:
                raise RuntimeError('duplicate/non-train/overlapping ReasonSeg image: ' + name)
            seen.add(name)
            annotation = annotations[name]
            if row['text'] != annotation['text'][0]:
                raise RuntimeError('public mirror query differs from original first query')
            image = data / 'reasonseg_images/train' / (name + '.jpg')
            if not image.exists() or image.read_bytes() != row['image']['bytes']:
                raise RuntimeError('cached ReasonSeg image bytes differ from pinned public release')
            with Image.open(image) as source_image:
                shape = (source_image.height, source_image.width)
            target, valid = annotation_masks(annotation, shape)
            mirror_target = np.asarray(row['mask'], dtype=bool)
            if mirror_target.shape != shape or not np.array_equal(target[valid], mirror_target[valid]):
                raise RuntimeError('public mirror target differs from original polygon on valid pixels')
            if not target.any():
                raise RuntimeError('empty positive original ReasonSeg annotation')
            identifier = hashlib.sha256(('reasonseg:' + name).encode()).hexdigest()[:24]
            target_path, valid_path = root / 'human_gt' / (identifier + '.png'), root / 'human_valid' / (identifier + '.png')
            target_path.parent.mkdir(parents=True, exist_ok=True)
            valid_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target.astype(np.uint8) * 255).save(target_path)
            Image.fromarray(valid.astype(np.uint8) * 255).save(valid_path)
            records.append({'id': identifier, 'dataset': 'reasonseg', 'image_id': name, 'image': str(image),
                'query': row['text'], 'source_split': 'train', 'private_no_target': False,
                'generated_query': False, 'pseudo_label': False, 'bucket': 'reason_positive',
                'holdout': int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 25 == 0,
                'human_gt_mask': str(target_path), 'human_valid_mask': str(valid_path),
                'original_annotation': str(root / 'original_annotations' / (name + '.json'))})
            if len(records) % 20 == 0:
                print(json.dumps({'status': 'VERIFYING_REASONSEG_ORIGINAL_MASKS', 'records': len(records)}), flush=True)
    if len(records) != 239 or len(seen) != len(annotations):
        raise RuntimeError('incomplete audited ReasonSeg train')
    report = {'source_only_train': True, 'generated_queries': False, 'pseudo_labels': False,
        'original_public_manifest_sha256': digest_file(Path(args.source_run) / 'TRAIN_MANIFEST.json'),
        'reasonseg_annotation_repo': ANNOTATION_REPO, 'reasonseg_annotation_revision': ANNOTATION_REVISION,
        'reasonseg_parquet_repo': PARQUET_REPO, 'reasonseg_parquet_revision': PARQUET_REVISION,
        'reasonseg_parquet_hashes': mirror_files, 'original_annotation_blob_hashes_verified': True,
        'published_text_includes_author_paraphrases': True, 'query_selection': 'existing first published query only',
        'ignore_regions_preserved': True, 'validation_test_images_excluded': True,
        'training_mixture': '10% ReasonSeg; 25% official gRefCOCO no-target; 65% original referring positives',
        'records': original['records'] + records, 'reasonseg_records': records}
    atomic_json(root / 'TRAIN_MANIFEST.json', report)
    digest = digest_file(root / 'TRAIN_MANIFEST.json')
    # Existing SAM features/human masks remain referenced in-place, never copied
    # or overwritten; only provenance records for this new mix are materialized.
    (root / 'cache/records').mkdir(parents=True, exist_ok=True)
    for old in (Path(args.source_run) / 'cache/records').glob('*.json'):
        value = json.loads(old.read_text())
        value['parent_source_manifest_sha256'] = value['source_manifest_sha256']
        value['source_manifest_sha256'] = digest
        atomic_json(root / 'cache/records' / old.name, value)
    atomic_json(root / 'SOURCE_AUDIT.json', {k: v for k, v in report.items() if k not in ('records', 'reasonseg_records')})
    print(json.dumps({'status': 'PUBLIC_REASONING_SOURCE_VERIFIED', 'reasonseg_train': len(records), 'output': str(root)}))


def encode(args):
    import torch
    from projects.evoseg.dialog.runtime import NativeRuntime, token_string
    from .cache_public_mix import cpu_tree
    root = Path(args.output)
    manifest = root / 'TRAIN_MANIFEST.json'
    source, digest = json.loads(manifest.read_text()), digest_file(manifest)
    runtime = NativeRuntime(args.model_dir, args.assets, load_language=False)
    for number, record in enumerate(source['reasonseg_records'], 1):
        destination = root / 'cache/records' / (record['id'] + '.json')
        if not destination.exists():
            runtime.set_image(record['image'])
            state_path = root / 'cache/sam_states' / (record['id'] + '.pth')
            state_path.parent.mkdir(parents=True, exist_ok=True)
            states = {k: runtime.sam_states[k] for k in ('current_vision_feats', 'feat_sizes')}
            torch.save(cpu_tree(states), state_path)
            with Image.open(record['human_gt_mask']) as mask:
                target = np.asarray(mask) > 0
            codes = runtime.encode(target)
            answer = '```json\n' + json.dumps([{'mask_2d': token_string(codes)}]) + '\n```'
            atomic_json(destination, {**record, 'sam_states': str(state_path), 'private_target_codes': codes,
                'assistant_target': answer, 'foundation_revision': runtime.revision,
                'source_manifest_sha256': digest, 'supervision': 'original published target polygons; ignore pixels excluded'})
        else:
            if json.loads(destination.read_text())['source_manifest_sha256'] != digest:
                raise RuntimeError('stale reasoning cache')
        atomic_json(root / 'CACHE_STATUS.json', {'records_done': number, 'records_total': 239, 'updated_at_unix': time.time()})
    atomic_json(root / 'CACHE_READY.json', {'records': len(list((root / 'cache/records').glob('*.json'))),
        'source_manifest_sha256': digest, 'generated_queries': False, 'pseudo_labels': False})


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default=base + '/datasets')
    p.add_argument('--source-run', default=base + '/results/evoseg_hallucination_20261007/new_public_run')
    p.add_argument('--output', default=base + '/results/evoseg_hallucination_20261007/reasoning_public_mix')
    p.add_argument('--model-dir', default=base + '/models/Qwen3-VL-8B-SAMTok-official')
    p.add_argument('--assets', default=base + '/results/evoseg_dialog_20261006/ASSETS.json')
    p.add_argument('--proxy', default='http://127.0.0.1:17890')
    p.add_argument('--encode', action='store_true')
    args = p.parse_args()
    encode(args) if args.encode else prepare(args)


if __name__ == '__main__':
    main()

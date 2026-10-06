"""Image-disjoint public train pilot split shared by training and diagnostics."""
import hashlib
import json
from pathlib import Path

DATASETS = ('refcoco', 'refcoco+', 'refcocog')
SPLIT_VERSION = 'public_cache_image_sha256_mod10_v1'


def is_holdout(image):
    return int(hashlib.sha256(Path(image).name.encode()).hexdigest()[:8], 16) % 10 == 0


def cache_records(cache, holdout=False):
    cache = Path(cache)
    manifest = json.loads((cache / 'MANIFEST.json').read_text())
    if (manifest['status'] != 'PUBLIC_TRAIN_PILOT_CACHE_READY' or
            manifest['generated_drafts_included'] or not manifest['validation_images_excluded']):
        raise RuntimeError('requires audited public-only cache excluding validation images')
    records = [json.loads(path.read_text()) for dataset in DATASETS
               for path in sorted((cache / dataset).glob('*.json'))]
    chosen = [record for record in records if is_holdout(record['image']) == holdout]
    if not chosen:
        raise RuntimeError('empty pilot split')
    return chosen


def heldout_source_records(cache, dataset, source_root):
    selected = [r for r in cache_records(cache, holdout=True) if r['dataset'] == dataset]
    path = Path(source_root) / f'mr_{dataset}_train.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(r['source_sha256'] != digest for r in selected):
        raise RuntimeError('public train source differs from audited cache')
    raw = json.loads(path.read_text())
    # Reconstruct queries from RAW markers, not GT-compiled training queries.
    return path, [raw[r['source_index']] for r in selected], digest

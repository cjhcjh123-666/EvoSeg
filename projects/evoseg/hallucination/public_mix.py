"""Original public queries and human masks; no generated query/pseudo-label data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import digest_file, public_pairs

OFFICIAL_GREF_REVISION = '81eede59b3ac070049f597d023c0ff08d1fb80e9'


def parquet_rows(path, columns=None):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def sample_id(dataset, ref_id, sent_id):
    return hashlib.sha256(f'{dataset}:{ref_id}:{sent_id}'.encode()).hexdigest()[:24]


def public_record(dataset, ref, sentence, images, annotation_source):
    ids = ref['ann_id'] if isinstance(ref['ann_id'], list) else [ref['ann_id']]
    empty = bool(ref.get('no_target', ids == [-1]))
    if empty != (ids == [-1]):
        raise ValueError('official no-target flag and target IDs disagree')
    image_id = int(ref['image_id'])
    image = Path(images) / f'COCO_train2014_{image_id:012d}.jpg'
    if not image.is_file():
        raise FileNotFoundError(image)
    return {'id': sample_id(dataset, ref['ref_id'], sentence['sent_id']), 'dataset': dataset,
            'ref_id': ref['ref_id'], 'sent_id': sentence['sent_id'], 'image_id': image_id,
            'image': str(image), 'query': sentence['sent'], 'source_split': ref['split'],
            'annotation_source': annotation_source, 'private_annotation_ids': ids,
            'private_no_target': empty, 'generated_query': False, 'pseudo_label': False}


def manifest(args):
    root = Path(args.gref_root)
    refs_path = root / 'grefs(unc).json'
    refs = json.loads(refs_path.read_text())
    official = json.loads((root / 'SOURCE.json').read_text())
    if official['revision'] != OFFICIAL_GREF_REVISION or not official['byte_hashes_verified']:
        raise RuntimeError('public source has not passed official-byte verification')
    forbidden = {int(r['image_id']) for r in refs if r['split'] != 'train'}
    sources = {'grefcoco': {'refs_sha256': digest_file(refs_path), 'revision': official['revision']}}
    for dataset in ('refcoco', 'refcocoplus', 'refcocog'):
        directory = Path(args.ref_root) / dataset
        for path in directory.glob('*.parquet'):
            if path.name != 'train.parquet':
                forbidden.update(int(row['image_id']) for row in parquet_rows(path, ['image_id']))
        sources[dataset] = {'train_parquet_sha256': digest_file(directory / 'train.parquet'),
                            'query_field': 'original sentences.sent', 'mask_validation': 'raw_anns vs original COCO'}
    # Explicit cross-benchmark image exclusion, not new annotations or texts.
    for pair in public_pairs(args.hallu_root):
        name = Path(pair['factual_image']).stem
        if name.startswith('COCO_train2014_'):
            forbidden.add(int(name.split('_')[2]))
    coco = json.loads(Path(args.coco_annotations).read_text())
    annotations = {int(a['id']): a for a in coco['annotations']}
    records, excluded = [], Counter()
    for ref in refs:
        if ref['split'] != 'train':
            continue
        if int(ref['image_id']) in forbidden:
            excluded['validation_or_test_image'] += len(ref['sentences'])
            continue
        records.extend(public_record('grefcoco', ref, sent, args.images, 'grefcoco') for sent in ref['sentences'])
    for dataset in ('refcoco', 'refcocoplus', 'refcocog'):
        for row in parquet_rows(Path(args.ref_root) / dataset / 'train.parquet'):
            if row['split'] != 'train':
                raise RuntimeError('non-train row in public training file')
            if int(row['image_id']) in forbidden:
                excluded['validation_or_test_image'] += len(row['sentences'])
                continue
            raw = json.loads(row['raw_anns'])
            annotation = annotations[int(row['ann_id'])]
            if raw['segmentation'] != annotation['segmentation'] or int(annotation['image_id']) != int(row['image_id']):
                raise RuntimeError('public RefCOCO export differs from original COCO annotation')
            records.extend(public_record(dataset, row, sent, args.images, 'coco') for sent in row['sentences'])
    # Deterministic bounded first-run corpus; per-image holdout spans all datasets.
    buckets = {'ref_positive': [], 'gref_positive': [], 'gref_empty': []}
    for record in sorted(records, key=lambda r: r['id']):
        bucket = ('gref_empty' if record['private_no_target'] else 'gref_positive') if record['dataset'] == 'grefcoco' else 'ref_positive'
        record['bucket'] = bucket
        record['holdout'] = int(hashlib.sha256(str(record['image_id']).encode()).hexdigest()[:8], 16) % 25 == 0
        buckets[bucket].append(record)
    selected = []
    for bucket, fraction in [('ref_positive', .5), ('gref_positive', .25), ('gref_empty', .25)]:
        available = buckets[bucket]
        wanted = int(args.max_records * fraction)
        if len(available) < wanted:
            raise RuntimeError(f'not enough audited public {bucket} records: {len(available)} < {wanted}')
        selected.extend(available[:wanted])
    selected.sort(key=lambda r: r['id'])
    train_images = {r['image_id'] for r in selected if not r['holdout']}
    holdout_images = {r['image_id'] for r in selected if r['holdout']}
    if train_images & holdout_images or (train_images | holdout_images) & forbidden:
        raise RuntimeError('image split leakage')
    report = {'status': 'PUBLIC_TRAIN_MANIFEST_READY', 'sources': sources, 'records': selected,
              'source_only_train': True, 'generated_queries': False, 'pseudo_labels': False,
              'halluseg_training_included': False, 'image_disjoint_holdout': True,
              'excluded_official_val_test_images': len(forbidden), 'exclusions': dict(excluded),
              'per_dataset': dict(Counter(r['dataset'] for r in selected)), 'total_records': len(selected),
              'train_records': sum(not r['holdout'] for r in selected),
              'holdout_records': sum(r['holdout'] for r in selected),
              'annotation_sources': {'grefcoco': str(root / 'instances.json'), 'coco': args.coco_annotations}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('do not overwrite a training manifest')
    atomic_json(path, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--gref-root', default=str(base / 'datasets/grefcoco-official-pinned'))
    p.add_argument('--ref-root', default=str(base / 'datasets/refcoco_hf'))
    p.add_argument('--images', default=str(base / 'datasets/coco2014/train2014'))
    p.add_argument('--coco-annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    p.add_argument('--hallu-root', default=str(base / 'datasets/hallusegbench/test'))
    p.add_argument('--max-records', type=int, default=16384)
    p.add_argument('--output', required=True)
    manifest(p.parse_args())


if __name__ == '__main__':
    main()

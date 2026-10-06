"""Fail closed on missing benchmark assets, report public split coverage."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from PIL import Image

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import SUBSETS, public_pairs, read_mask, digest_file


def audit_pair(pair):
    values = {'subset': pair['subset'], 'index': pair['index']}
    for kind in ('factual', 'counterfactual'):
        with Image.open(pair[kind + '_image']) as image:
            image.verify()
        mask = read_mask(pair[kind + '_mask'])
        values[kind + '_gt_empty'] = not mask.any()
    return values


def audit(args):
    pairs = public_pairs(args.data_root)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(audit_pair, pairs))
    counts = Counter(p['subset'] for p in pairs)
    if counts != Counter(SUBSETS):
        raise RuntimeError('public HalluSegBench coverage differs')
    refs = json.loads(Path(args.grefs).read_text())
    splits = {}
    for split in ('train', 'val', 'testA', 'testB'):
        selected = [r for r in refs if r['split'] == split]
        splits[split] = {'refs': len(selected), 'expressions': sum(len(r['sentences']) for r in selected),
                        'no_target_expressions': sum(len(r['sentences']) for r in selected
                                                    if r['no_target'] or r['ann_id'] == [-1])}
        for record in selected:
            if not (Path(args.coco_images) / record['file_name']).is_file():
                raise FileNotFoundError('missing gRefCOCO image: ' + record['file_name'])
    report = {'status': 'PUBLIC_BENCHMARK_ASSETS_AUDITED', 'halluseg_pairs': len(pairs),
              'halluseg_predictions_per_model': len(pairs) * 4, 'subsets': dict(counts),
              'empty_gt_masks': sum(int(r[k]) for r in checked for k in ('factual_gt_empty', 'counterfactual_gt_empty')),
              'grefcoco_splits': splits, 'grefs_sha256': digest_file(args.grefs),
              'training_started': False, 'sota_claimed': False,
              'halluseg_train_available': (Path(args.data_root).parent / 'train').is_dir()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--data-root', default=str(base / 'datasets/hallusegbench/test'))
    p.add_argument('--grefs', default=str(base / 'datasets/grefcoco/grefs_unc.json'))
    p.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    p.add_argument('--output', required=True)
    audit(p.parse_args())


if __name__ == '__main__':
    main()

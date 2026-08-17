#!/usr/bin/env python
"""Build counterfactual (q+, q-) pairs from gRefCOCO train for EvoSeg GRPO v2.

For every present-target train referring expression q+ (image V, GT mask), we
construct a counterfactual q- on the SAME image: "the {category}" where
{category} is a COCO category that has NO instance annotated in V. The model
must segment q+ and abstain on q- -> this is the per-image selective-prediction
signal (counterfactual pair reward).

Output (json list), one entry per present-train ref:
  {file, query, image_id, ann_ids, absent_names: [...]}

The GRPO trainer loads this manifest directly (or builds it on the fly with the
same logic). The manifest is also reusable for the image side of the video
faithfulness benchmark's "Counterfactual Query Swap" category.

Usage:
  python projects/evoseg/tools/build_counterfactual_pairs.py \
      --out /9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/train_counterfactual_pairs.json \
      [--max-refs 2000]
"""
import argparse
import json
import os

GREFS = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/grefs_unc.json'
INSTANCES = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/instances.json'
IMG_ROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/'
            'glamm_data/images/coco2014/train2014')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='output json path')
    p.add_argument('--max-refs', type=int, default=None, help='cap #pairs (debug)')
    p.add_argument('--max-absents', type=int, default=2,
                   help='max absent categories kept per ref (random subset)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    grefs = json.load(open(GREFS))
    instances = json.load(open(INSTANCES))
    anns = instances['annotations']
    cat_names = {c['id']: c['name'] for c in instances['categories']}

    # category ids present per image (from annotated instances)
    img_cats = {}
    for a in anns:
        img_cats.setdefault(a['image_id'], set()).add(a['category_id'])
    all_cat_ids = set(cat_names)

    import random
    rng = random.Random(args.seed)
    out = []
    n_skip = 0
    for r in grefs:
        if r['split'] != 'train' or r['no_target'] or r['ann_id'] == [-1]:
            continue
        fn = os.path.join(IMG_ROOT, r['file_name'])
        if not os.path.exists(fn):
            continue
        iid = r['image_id']
        present_cats = img_cats.get(iid, set())
        absent_cats = sorted(all_cat_ids - present_cats)
        # drop categories whose name appears in the query (avoid near-duplicate
        # q- that might actually match the referent's category)
        q = r['sentences'][0]['sent'].lower()
        absent_cats = [c for c in absent_cats if cat_names[c].lower() not in q]
        if not absent_cats:
            n_skip += 1
            continue
        rng.shuffle(absent_cats)
        absent_cats = absent_cats[:args.max_absents]
        out.append({
            'file': fn,
            'query': r['sentences'][0]['sent'],
            'image_id': iid,
            'ann_ids': r['ann_id'],
            'absent_names': [cat_names[c] for c in absent_cats],
        })
        if args.max_refs and len(out) >= args.max_refs:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), ensure_ascii=False, indent=1)
    print(f'wrote {len(out)} counterfactual pairs (skipped {n_skip} no-absent-cat) '
          f'-> {args.out}')


if __name__ == '__main__':
    main()

"""Failure taxonomy for remaining image hallucinations.

The Faithful model still hallucinates on a subset of absent queries (1309/8905).
This classifies each failure by query structure and by whether the query's
category is actually present in the image (near-miss lookalike) or completely
absent (clear false premise / far-miss).

Taxonomy:
  short_noun  : <=3 words, no attribute/relation (e.g. "the white dog")
  attr_rich   : >3 words with attribute/relation modifiers
  lookalike   : the COCO category named in the query IS present in the image
                (wrong instance / wrong attribute / wrong position -> hard)
  far_miss    : no queried category present at all (clear false premise)

This separates "genuinely hard near-misses" from "clear false premises" so the
paper can honestly report what the remaining 14.7% hallucination is made of.
"""
import argparse
import collections
import json
import re

GREFS = ('/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/'
         'grefs_unc.json')
COCO_ANN = ('/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/'
            'instances.json')


def load_coco():
    coco = json.load(open(COCO_ANN))
    cat_id2name = {c['id']: c['name'] for c in coco['categories']}
    img_cats = collections.defaultdict(set)  # image_id -> set of cat names
    for ann in coco['annotations']:
        img_cats[ann['image_id']].add(cat_id2name[ann['category_id']])
    return cat_id2name, img_cats


def image_id_from_file(fn):
    m = re.search(r'(\d+)\.(jpg|png)$', fn)
    return int(m.group(1)) if m else None


def query_categories(sent, cat_names):
    s = ' ' + sent.lower() + ' '
    hits = []
    for name in cat_names:
        # word-boundary substring match, longest names first
        if re.search(r'\b' + re.escape(name) + r'\b', s):
            hits.append(name)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--halluc-json', required=True,
                    help='eval json with results[].n_masks/sent/file/pred_text')
    ap.add_argument('--model-name', default='model')
    args = ap.parse_args()

    cat_id2name, img_cats = load_coco()
    cat_names = sorted(set(cat_id2name.values()), key=len, reverse=True)
    d = json.load(open(args.halluc_json))
    res = d['results']
    hall = [r for r in res if r.get('n_masks', 0) > 0]
    tot = len(hall)
    print(f'[{args.model_name}] hallucinated: {tot} / {len(res)}')
    if tot == 0:
        return

    tax = collections.Counter()
    lookalike = collections.Counter()   # lookalike vs far_miss
    examples = {'lookalike': [], 'far_miss': []}
    for r in hall:
        sent = r['sent']
        nw = len(sent.split())
        qcats = query_categories(sent, cat_names)
        img_id = image_id_from_file(r['file'])
        present = img_cats.get(img_id, set())
        is_lookalike = bool(set(qcats) & present)
        key = 'short_noun' if nw <= 3 else 'attr_rich'
        tax[key] += 1
        lookalike['lookalike' if is_lookalike else 'far_miss'] += 1
        grp = 'lookalike' if is_lookalike else 'far_miss'
        if len(examples[grp]) < 12:
            examples[grp].append((sent, r['file'], ','.join(qcats),
                                  sorted(present)[:6]))

    print('\n=== query-structure taxonomy ===')
    for k in ('short_noun', 'attr_rich'):
        print(f'  {k}: {tax[k]} ({tax[k]/tot:.1%})')
    print('\n=== grounding hardness ===')
    for k in ('lookalike', 'far_miss'):
        print(f'  {k}: {lookalike[k]} ({lookalike[k]/tot:.1%})')
    # joint table
    print('\n=== joint (structure x hardness) ===')
    joint = collections.Counter()
    for r in hall:
        nw = len(r['sent'].split())
        qcats = query_categories(r['sent'], cat_names)
        present = img_cats.get(image_id_from_file(r['file']), set())
        lk = bool(set(qcats) & present)
        joint[(('short' if nw <= 3 else 'attr'),
               ('lookalike' if lk else 'far_miss'))] += 1
    for k in sorted(joint):
        print(f'  {k[0]:5s} x {k[1]:9s}: {joint[k]} ({joint[k]/tot:.1%})')

    print('\n--- lookalike (near-miss) examples ---')
    for s, f, qc, pc in examples['lookalike']:
        print(f'  [{qc}] "{s}"  img={f}  present={pc}')
    print('\n--- far-miss examples ---')
    for s, f, qc, pc in examples['far_miss']:
        print(f'  [{qc}] "{s}"  img={f}  present={pc}')


if __name__ == '__main__':
    main()

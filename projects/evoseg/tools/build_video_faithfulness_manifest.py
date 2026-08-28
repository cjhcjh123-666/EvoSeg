#!/usr/bin/env python
"""Build the EvoSeg video faithfulness benchmark from Ref-YT-VOS valid.

The standard RVOS evaluation assumes the reference is valid everywhere. This
benchmark adds the faithfulness dimensions that general RVOS does not test:

  temporal_absence   : original expression whose referent exists only on a
                       sub-interval [t0, t1] (appears late and/or disappears
                       early). Expected: masks exactly on the presence window;
                       after t1 the model must STOP (SAM2 memory hallucination).
  global_absence     : generated "the {cat}" where {cat} has no instance in the
                       whole video. Expected: empty on every frame.
  counterfactual_swap: original expression with its referent category word
                       replaced by an absent category. Expected: empty.
  identity_swap      : original expression in a video that contains >=2
                       objects of the same category (lookalike instances).
                       Expected: segment the referenced instance only, never
                       jump to the lookalike.

Every case records `presence` (expected mask-presence per frame, derived from
GT for real expressions / all-False for generated ones) so a faithfulness eval
harness can compute per-frame mask-vs-presence agreement (hallucinated frames
on absent intervals, missed frames on present intervals) without needing the
reference masks at eval time.

Usage:
  python projects/evoseg/tools/build_video_faithfulness_manifest.py \
      --out /9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_valid.json
"""
import argparse
import json
import os
import random
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

META = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
        'extracted/valid/meta_expressions_challenge.json')
ANNROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
           'extracted/valid/Annotations')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-videos', type=int, default=None)
    return p.parse_args()


def video_presence(args):
    """Compute per-frame presence (GT mask non-empty) for every expression of
    one video, aligned to the video's frame list."""
    vid, v, annroot = args
    frames = v['frames']
    fidx = {f: i for i, f in enumerate(frames)}
    n_frames = len(frames)
    exps = {}
    for exp_id, e in v['expressions'].items():
        d = os.path.join(annroot, vid, exp_id)
        if not os.path.isdir(d):
            continue
        has = np.zeros(n_frames, dtype=bool)
        for fr in sorted(os.listdir(d)):
            if not fr.endswith('.png'):
                continue
            i = fidx.get(fr[:-4])
            if i is None:
                continue
            m = np.array(Image.open(os.path.join(d, fr)).convert('L')) > 0
            has[i] = bool(m.sum())
        exps[exp_id] = {'exp': e['exp'], 'obj_id': e['obj_id'], 'presence': has}
    return vid, v, exps


def main():
    args = parse_args()
    random.seed(0)
    meta = json.load(open(META))
    videos = meta['videos']
    if args.max_videos:
        videos = {k: v for i, (k, v) in enumerate(videos.items())
                  if i < args.max_videos}

    tasks = [(vid, v, ANNROOT) for vid, v in videos.items()]
    cases = []
    stats = Counter()
    with Pool(processes=args.workers) as pool:
        for i, (vid, v, exps) in enumerate(pool.imap_unordered(video_presence, tasks)):
            frames = v['frames']
            T = len(frames)
            # category per object
            obj_cat = {oid: o.get('category', '') for oid, o in v.get('objects', {}).items()}
            cat_set = set(obj_cat.values())
            # absent COCO-ish categories: hand-picked common nouns not present
            absent_cats = [c for c in
                           ('elephant', 'giraffe', 'zebra', 'piano', 'airplane',
                            'surfboard', 'skateboard', 'motorcycle', 'horse',
                            'cow', 'sheep', 'backpack', 'suitcase', 'umbrella',
                            'handbag', 'tennis racket', 'baseball bat', 'frisbee')
                           if c not in cat_set]
            if not absent_cats:
                absent_cats = ['elephant']

            # identity-swap videos: >=2 objects share a category
            cat_cnt = Counter(obj_cat.values())
            multi_cats = {c for c, n in cat_cnt.items() if n >= 2}

            for exp_id, info in exps.items():
                has = info['presence']
                idx = np.where(has)[0]
                if len(idx) == 0:
                    continue
                t0, t1 = int(idx[0]), int(idx[-1])
                obj = obj_cat.get(info['obj_id'], '')

                # 1) temporal absence (appears late and/or disappears early)
                if t0 > 0 or t1 < T - 1:
                    aspects = []
                    if t1 < T - 1:
                        aspects.append('disappear_early')
                    if t0 > 0:
                        aspects.append('appear_late')
                    for asp in aspects:
                        cases.append({
                            'category': 'temporal_absence',
                            'aspect': asp,
                            'video_id': vid, 'exp_id': exp_id,
                            'obj_id': info['obj_id'], 'query': info['exp'],
                            'frames': frames, 'interval': [t0, t1],
                            'presence': has.tolist(), 'src_exp_id': None,
                        })
                        stats['temporal_absence'] += 1

                # 2) global absence: a RANDOM absent category (not always
                #    'elephant' — avoids category imbalance in negatives)
                q = f'the {random.choice(absent_cats)}'
                cases.append({
                    'category': 'global_absence', 'aspect': None,
                    'video_id': vid, 'exp_id': None, 'obj_id': None,
                    'query': q, 'frames': frames, 'interval': None,
                    'presence': [False] * T, 'src_exp_id': exp_id,
                })
                stats['global_absence'] += 1

                # 3) counterfactual swap: replace referent category word with
                #    a RANDOM absent category
                if obj and obj.lower() in info['exp'].lower():
                    q2 = replace_first(info['exp'], obj, random.choice(absent_cats))
                    cases.append({
                        'category': 'counterfactual_swap', 'aspect': None,
                        'video_id': vid, 'exp_id': None, 'obj_id': None,
                        'query': q2, 'frames': frames, 'interval': None,
                        'presence': [False] * T, 'src_exp_id': exp_id,
                    })
                    stats['counterfactual_swap'] += 1

                # 4) identity swap: lookalike instances of the same category
                if obj in multi_cats:
                    cases.append({
                        'category': 'identity_swap', 'aspect': None,
                        'video_id': vid, 'exp_id': exp_id,
                        'obj_id': info['obj_id'], 'query': info['exp'],
                        'frames': frames, 'interval': [t0, t1],
                        'presence': has.tolist(), 'src_exp_id': None,
                    })
                    stats['identity_swap'] += 1

            if (i + 1) % 40 == 0:
                print(f'  {i + 1}/{len(tasks)} videos', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {'n_videos': len(videos), 'cases': cases, 'category_counts': dict(stats)}
    json.dump(out, open(args.out, 'w'), ensure_ascii=False)
    print('category_counts:', dict(stats))
    print(f'wrote {len(cases)} faithfulness cases -> {args.out}')


def replace_first(text, old, new):
    """Case-insensitive first-occurrence replacement of `old` by `new`."""
    low = text.lower()
    i = low.find(old.lower())
    if i < 0:
        return text
    return text[:i] + new + text[i + len(old):]


if __name__ == '__main__':
    main()

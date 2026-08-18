#!/usr/bin/env python
"""Build the EvoSeg video faithfulness TRAINING manifest from Ref-YT-VOS train.

Ref-YT-VOS train stores per-video indexed instance masks (pixel value = instance
id), unlike valid which has per-expression annotation dirs. This builder
derives per-frame target presence from the indexed masks and emits the same
faithfulness taxonomy used by the eval benchmark, so the model can be trained
end-to-end on the exact failure modes RVOS ignores:

  temporal_absence   : real expression whose referent exists only on a strict
                       sub-interval (appears late and/or disappears early).
                       The mask must STOP after disappearance (SAM2 memory
                       hallucination) -> GT masks are empty on absent frames.
  identity_swap      : video with >=2 co-occurring instances (lookalikes);
                       expression references one instance -> the mask must not
                       jump to the lookalike (GT = referenced instance masks).
  global_absence     : mismatched (cross-video) query that refers to nothing in
                       this video -> GT empty on every frame, model must refuse.

Every case records `presence` (expected mask-presence per frame) and `instance`
(the referenced instance id, None for generated negatives) so the dataset can
load per-frame GT masks directly from the indexed annotation PNGs.

Usage:
  python projects/evoseg/tools/build_video_faithfulness_train.py \
      --out /9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_train.json
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
        'extracted/meta_expressions/train/meta_expressions.json')
ANNROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
           'extracted/train/Annotations')

SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-videos', type=int, default=None)
    p.add_argument('--neg-per-video', type=int, default=2,
                   help='cross-video negative queries per video (global_absence)')
    return p.parse_args()


def instance_order_and_presence(vid, frames, annroot):
    """First-appearance instance order + per-instance per-frame presence."""
    vid_dir = os.path.join(annroot, vid)
    seen = []
    present = {}
    for frame in frames:
        png = os.path.join(vid_dir, frame + '.png')
        if not os.path.isfile(png):
            continue
        a = np.asarray(Image.open(png).convert('L'))
        vals = set(np.unique(a).tolist())
        vals.discard(0)
        for v in vals:
            if v not in present:
                present[v] = set()
                seen.append(v)
            present[v].add(frame)
    return seen, present


def process_video(args):
    vid, v, annroot, negs = args
    frames = sorted(v['frames'])
    T = len(frames)
    if T < 4:
        return vid, []
    order, present = instance_order_and_presence(vid, frames, annroot)
    inst_presence = {inst: [frame in present.get(inst, set()) for frame in frames]
                     for inst in order}
    n_inst = len(order)
    cases = []

    for exp_id, info in v['expressions'].items():
        exp = info['exp']
        obj_idx = int(info['obj_id'])
        if not (0 < obj_idx <= n_inst):
            continue
        inst = order[obj_idx - 1]
        has = np.array(inst_presence[inst], dtype=bool)
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        t0, t1 = int(idx[0]), int(idx[-1])

        # 1) temporal absence: appears late and/or disappears early
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
                    'obj_id': obj_idx, 'instance': inst,
                    'query': exp, 'frames': frames,
                    'interval': [t0, t1],
                    'presence': has.tolist(),
                })

        # 2) identity swap: >=2 instances co-occurring in this video
        if n_inst >= 2:
            cases.append({
                'category': 'identity_swap',
                'aspect': None,
                'video_id': vid, 'exp_id': exp_id,
                'obj_id': obj_idx, 'instance': inst,
                'query': exp, 'frames': frames,
                'interval': [t0, t1],
                'presence': has.tolist(),
            })

    # 3) global absence: cross-video mismatched queries (verifiable-ish hard
    #    negatives). Sampling outside the pool so negatives are not the same
    #    video's own expressions; textual similarity is kept low.
    for q in negs:
        cases.append({
            'category': 'global_absence',
            'aspect': None,
            'video_id': vid, 'exp_id': None,
            'obj_id': None, 'instance': None,
            'query': q, 'frames': frames,
            'interval': None,
            'presence': [False] * T,
        })
    return vid, cases


def main():
    args = parse_args()
    random.seed(SEED)
    meta = json.load(open(META))
    videos = meta['videos']
    if args.max_videos:
        videos = {k: v for i, (k, v) in enumerate(videos.items())
                  if i < args.max_videos}

    # pool of expressions for cross-video negatives (exclude the video itself)
    all_exps = []
    for vn, vd in videos.items():
        for e in vd['expressions'].values():
            all_exps.append(e['exp'])
    all_exps = [e for e in all_exps if 4 <= len(e.split()) <= 24]

    tasks = []
    for vn, vd in videos.items():
        own = {e['exp'] for e in vd['expressions'].values()}
        pool = [e for e in all_exps if e not in own]
        negs = random.sample(pool, min(args.neg_per_video, len(pool)))
        tasks.append((vn, vd, ANNROOT, negs))

    cases = []
    stats = Counter()
    with Pool(processes=args.workers) as pool:
        for i, (vid, vcases) in enumerate(pool.imap_unordered(process_video, tasks)):
            for c in vcases:
                cases.append(c)
                stats[c['category']] += 1
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(tasks)} videos', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {'n_videos': len(videos), 'cases': cases,
           'category_counts': dict(stats)}
    json.dump(out, open(args.out, 'w'), ensure_ascii=False)
    print('category_counts:', dict(stats))
    print(f'wrote {len(cases)} faithfulness train cases -> {args.out}')


if __name__ == '__main__':
    main()

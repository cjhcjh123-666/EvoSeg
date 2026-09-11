"""M5.1 — mine a high-quality ambiguous-referent pilot pool.

Uses ONLY real Ref-YT-VOS valid expressions and GT instance tracks (no toy
negatives such as random absent classes or category-word swaps).

Signals (each independent; baseline failure is optional and never the sole
criterion):

  n_same_cat              # other instances sharing the target's category
  n_other                 # other instances in the video
  co_visible_frames/ratio frames where target AND a same-category distractor are both present
  min_centroid_dist_norm  closest normalised centroid distance (target vs best distractor)
  max_dilated_overlap     largest dilated-overlap between target and distractor
  target_gap_frames       target absent in the middle then returns (reappearance)
  target_absent_frames    target absent frames total
  query_type              dataset-derived vocabulary: motion / temporal_order / relation / appearance / other

Ranking is a transparent z-score composite over the mining signals; the
baseline-IDErr signal is written as a separate column and is NOT required.

Outputs
-------
<candidate_pool.json>   full pool (>=150-250 cases) with stable keys
<candidate_summary.csv> flat summary for inspection
<candidates.html>       local contact sheet (no server required)
<stats.json>            dataset-level statistics / distributions
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

DEFAULT_ANN = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/Annotations'
DEFAULT_META = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/meta_expressions_challenge.json'
DEFAULT_JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'

# ---------------------------------------------------------------- vocabulary
MOTION_WORDS = {
    'moving', 'walks', 'walking', 'runs', 'running', 'jumps', 'jumping', 'flies', 'flying',
    'swims', 'swimming', 'rides', 'riding', 'dances', 'dancing', 'turns', 'turning', 'goes',
    'going', 'comes', 'coming', 'drives', 'driving', 'slides', 'sliding', 'chases', 'throwing',
    'catch', 'catches', 'kicks', 'pushing', 'pulls', 'lifts', 'climbs',
}
TEMPORAL_ORDER_WORDS = {
    'first', 'second', 'third', 'last', 'then', 'after', 'before', 'next', 'finally',
    'initially', 'later', 'follows', 'followed', 'following', 'again', 'returns', 'returning',
    'reappears', 'starts', 'stops',
}
RELATION_WORDS = {
    'behind', 'front', 'left', 'right', 'next', 'near', 'beside', 'above', 'below', 'under',
    'between', 'on', 'in', 'with', 'closest', 'farthest', 'top', 'bottom', 'middle', 'center',
    'carrying', 'wearing', 'holding', 'touching', 'pushing', 'sitting', 'standing', 'lying',
}
APPEARANCE_WORDS = {
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'gray', 'grey', 'orange',
    'pink', 'purple', 'dark', 'light', 'bright', 'large', 'small', 'big', 'little', 'tall',
    'short', 'striped', 'colorful', 'wooden', 'metal', 'plastic',
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--meta', default=DEFAULT_META)
    ap.add_argument('--ann-root', default=DEFAULT_ANN)
    ap.add_argument('--jpeg-root', default=DEFAULT_JPEG)
    ap.add_argument('--out-dir', default='docs/premature_commitment/results')
    ap.add_argument('--pool-size', type=int, default=250)
    ap.add_argument('--top-k', type=int, default=80)
    ap.add_argument('--max-videos', type=int, default=None, help='debug: cap #videos')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def load_masks(vid, exp_id, frames, ann_root):
    """[(frame, mask|None)] using the expression's own annotation dir."""
    out = []
    for f in frames:
        p = os.path.join(ann_root, vid, str(exp_id), f + '.png')
        if os.path.exists(p):
            out.append(np.array(Image.open(p).convert('L')) > 0)
        else:
            out.append(None)
    return out


def centroid_and_area(m):
    if m is None or not m.any():
        return None, 0.0
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return (float(xs.mean()) / w, float(ys.mean()) / h), float(m.mean())


def query_type(q):
    toks = set(re.findall(r"[a-z']+", q.lower()))
    scores = {
        'motion': len(toks & MOTION_WORDS),
        'temporal_order': len(toks & TEMPORAL_ORDER_WORDS),
        'relation': len(toks & RELATION_WORDS),
        'appearance': len(toks & APPEARANCE_WORDS),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return 'other', scores
    if scores[best] >= 2:
        return 'multi_' + best, scores
    return best, scores


def main():
    args = parse_args()
    # The mining + ranking pipeline has no stochastic step, so results are
    # bit-identical across runs; --seed exists only so every M5 entry point
    # accepts the same interface. Say so instead of silently ignoring it.
    print('[mine] deterministic pipeline; --seed is accepted but unused '
          '(no stochastic step exists in mining/ranking)', flush=True)
    meta = json.load(open(args.meta))['videos']
    vids = list(meta.keys())
    if args.max_videos:
        vids = vids[:args.max_videos]
    os.makedirs(args.out_dir, exist_ok=True)

    cases = []
    for vid in vids:
        v = meta[vid]
        objs = v.get('objects') or {}
        if not objs:
            continue
        cat_of = {str(o): d.get('category', '') for o, d in objs.items()}
        for exp_id, e in v['expressions'].items():
            tgt = str(e['obj_id'])
            frames = v['frames']
            tgt_masks = load_masks(vid, exp_id, frames, args.ann_root)
            tgt_pres = np.array([m is not None and m.any() for m in tgt_masks])
            if not tgt_pres.any():
                continue  # expression with no annotated frames
            others = [o for o in cat_of if o != tgt]
            same_cat = [o for o in others if cat_of[o] == cat_of[tgt]]
            # masks of distractor instances: reach via any expression of that obj
            exp_of_obj = {}
            for eid, ee in v['expressions'].items():
                exp_of_obj.setdefault(str(ee['obj_id']), str(eid))
            dist_masks = {}
            for o in others:
                if o in exp_of_obj:
                    dist_masks[o] = load_masks(vid, exp_of_obj[o], frames, args.ann_root)

            co_visible = 0            # frames where target AND >=1 same-cat distractor are both present
            min_cdist = None
            max_adj = 0.0             # dilated-overlap proxy (instance masks never overlap by construction)
            for t in range(len(frames)):
                tm = tgt_masks[t]
                if tm is None or not tm.any():
                    continue
                tc, ta = centroid_and_area(tm)
                seen_this_frame = False
                for o in same_cat:
                    dm = dist_masks.get(o, [None] * len(frames))[t]
                    if dm is None or not dm.any():
                        continue
                    seen_this_frame = True
                    dc, da = centroid_and_area(dm)
                    d = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))
                    min_cdist = d if min_cdist is None else min(min_cdist, d)
                    # dilation radius ~2% of the shorter side
                    r = max(1, int(0.02 * min(tm.shape)))
                    tmd = ndimage.binary_dilation(tm, iterations=r)
                    inter = np.logical_and(tmd, dm).sum()
                    union = np.logical_or(tmd, dm).sum()
                    if union:
                        max_adj = max(max_adj, float(inter / union))
                if seen_this_frame:
                    co_visible += 1
            # gaps / reappearance
            pres = tgt_pres.astype(int)
            idx = np.where(pres == 1)[0]
            gap = 0
            if len(idx) > 1:
                gap = int(sum(1 for i in range(idx[0], idx[-1]) if pres[i] == 0))
            absent = int((pres == 0).sum())
            qt, qt_scores = query_type(e['exp'])
            n_tgt_frames = int(pres.sum())
            cases.append({
                'video_id': vid, 'exp_id': str(exp_id), 'query': e['exp'],
                'target_obj_id': tgt, 'candidate_obj_ids': sorted(others),
                'same_category_obj_ids': sorted(same_cat),
                'annotated_frames': frames,
                'target_category': cat_of[tgt],
                'mining_signals': {
                    'n_same_cat': len(same_cat),
                    'n_other': len(others),
                    'co_visible_frames': co_visible,
                    'co_visible_ratio': (co_visible / n_tgt_frames) if n_tgt_frames else 0.0,
                    'min_centroid_dist_norm': min_cdist,
                    'max_dilated_overlap': max_adj,
                    'target_gap_frames': gap,
                    'target_absent_frames': absent,
                    'query_type': qt,
                    'query_type_scores': qt_scores,
                    'n_target_frames': n_tgt_frames,
                    'n_frames': len(frames),
                },
            })
    if not cases:
        raise RuntimeError('no cases mined — check paths')

    # ------------------------------------------------------------- scoring
    def z(vals):
        a = np.asarray(vals, dtype=float)
        a = np.nan_to_num(a, nan=0.0)
        s = a.std()
        return (a - a.mean()) / s if s > 0 else np.zeros_like(a)

    sig = [c['mining_signals'] for c in cases]
    z_co = z([s['co_visible_ratio'] for s in sig])
    z_close = z([-(s['min_centroid_dist_norm'] if s['min_centroid_dist_norm'] is not None else 1.0) for s in sig])
    z_ov = z([s['max_dilated_overlap'] for s in sig])
    z_sc = z([s['n_same_cat'] for s in sig])
    z_gap = z([s['target_gap_frames'] for s in sig])
    z_qt = z([1.0 if s['query_type'] not in ('other', 'appearance') else 0.0 for s in sig])
    for i, c in enumerate(cases):
        c['mining_signals']['rank_score'] = float(
            1.0 * z_co[i] + 0.8 * z_close[i] + 0.8 * z_ov[i] + 0.6 * z_sc[i] + 0.5 * z_gap[i] + 0.4 * z_qt[i])
    cases.sort(key=lambda c: -c['mining_signals']['rank_score'])
    pool = cases[:args.pool_size]
    # stratified top-k: soft quotas over query_type so the annotation pool is not
    # dominated by a single query family (proportional to the mined distribution,
    # with a floor of 3 per family present in the pool).
    fam = [c['mining_signals']['query_type'] for c in pool]
    counts = Counter(fam)
    total_pool = len(pool)
    quota = {f: max(3, int(round(args.top_k * n / total_pool))) for f, n in counts.items()}
    top, used = [], Counter()
    for c in pool:
        f = c['mining_signals']['query_type']
        if used[f] < quota.get(f, 0):
            top.append(c); used[f] += 1
        if len(top) >= args.top_k:
            break
    if len(top) < args.top_k:                      # backfill by rank
        chosen = {id(c) for c in top}
        for c in pool:
            if id(c) not in chosen:
                top.append(c)
            if len(top) >= args.top_k:
                break

    json.dump({'n_cases': len(pool), 'top_k': len(top),
               'ranking': 'z-score composite over mining signals (baseline failure NOT required)',
               'cases': pool},
              open(os.path.join(args.out_dir, 'candidate_pool.json'), 'w'), ensure_ascii=False)

    # summary CSV
    cols = ['video_id', 'exp_id', 'target_obj_id', 'target_category', 'query',
            'n_same_cat', 'n_other', 'co_visible_frames', 'co_visible_ratio',
            'min_centroid_dist_norm', 'max_dilated_overlap', 'target_gap_frames',
            'target_absent_frames', 'query_type', 'rank_score', 'in_top_k']
    with open(os.path.join(args.out_dir, 'candidate_summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(cols)
        topkeys = {f"{c['video_id']}:{c['exp_id']}" for c in top}
        for c in pool:
            s = c['mining_signals']
            w.writerow([c['video_id'], c['exp_id'], c['target_obj_id'], c['target_category'],
                        c['query'], s['n_same_cat'], s['n_other'], s['co_visible_frames'],
                        round(s['co_visible_ratio'], 4),
                        None if s['min_centroid_dist_norm'] is None else round(s['min_centroid_dist_norm'], 4),
                        round(s['max_dilated_overlap'], 4), s['target_gap_frames'],
                        s['target_absent_frames'], s['query_type'], round(s['rank_score'], 4),
                        int(f"{c['video_id']}:{c['exp_id']}" in topkeys)])

    # ------------------------------------------------------------- stats
    psig = [c['mining_signals'] for c in pool]
    tsig = [c['mining_signals'] for c in top]
    dist = {k: dict(Counter(s[k] for s in psig)) for k in ['query_type', 'n_same_cat']}
    stats = {
        'n_cases_total_mined': len(cases), 'pool_size': len(pool), 'top_k': len(top),
        'pool_co_visible_frames_median': float(np.median([s['co_visible_frames'] for s in psig])),
        'pool_co_visible_ratio_median': float(np.median([s['co_visible_ratio'] for s in psig])),
        'pool_n_same_cat_ge2_frac': float(np.mean([s['n_same_cat'] >= 2 for s in psig])),
        'topk_co_visible_frames_median': float(np.median([s['co_visible_frames'] for s in tsig])),
        'topk_n_same_cat_ge2_frac': float(np.mean([s['n_same_cat'] >= 2 for s in tsig])),
        'topk_query_type_distribution': dict(Counter(s['query_type'] for s in tsig)),
        'videos': len({c['video_id'] for c in pool}),
        'target_category_top20': dict(Counter(c['target_category'] for c in pool).most_common(20)),
        'query_type_distribution': dist['query_type'],
        'n_same_cat_distribution': dist['n_same_cat'],
        # NOTE: the three blocks below are over ALL MINED cases (n_cases_total_mined),
        # not over the pool; the `pool_*` keys above are the pool-scoped ones.
        'mined_co_visible_frames': {
            'mean': float(np.mean([s['co_visible_frames'] for s in sig])),
            'median': float(np.median([s['co_visible_frames'] for s in sig])),
            'max': int(np.max([s['co_visible_frames'] for s in sig])),
        },
        'mined_co_visible_ratio': {
            'mean': float(np.mean([s['co_visible_ratio'] for s in sig])),
            'median': float(np.median([s['co_visible_ratio'] for s in sig])),
            'max': float(np.max([s['co_visible_ratio'] for s in sig])),
        },
        'mined_has_reappearance_gap': int(sum(1 for s in sig if s['target_gap_frames'] > 0)),
        'mined_has_absent_frames': int(sum(1 for s in sig if s['target_absent_frames'] > 0)),
        'mined_has_dilated_overlap': int(sum(1 for s in sig if s['max_dilated_overlap'] > 0)),
        'pool_has_reappearance_gap': int(sum(1 for s in psig if s['target_gap_frames'] > 0)),
        'pool_has_absent_frames': int(sum(1 for s in psig if s['target_absent_frames'] > 0)),
        'pool_has_dilated_overlap': int(sum(1 for s in psig if s['max_dilated_overlap'] > 0)),
        'topk_has_reappearance_gap': int(sum(1 for s in tsig if s['target_gap_frames'] > 0)),
        'topk_has_absent_frames': int(sum(1 for s in tsig if s['target_absent_frames'] > 0)),
        'topk_has_dilated_overlap': int(sum(1 for s in tsig if s['max_dilated_overlap'] > 0)),
        'strata_in_top_k': dict(Counter(c['mining_signals']['query_type'] for c in top)),
        'note': 'pool is stratified across query types; baseline failure is a separate optional '
                'column; keys prefixed mined_/pool_/topk_ state their scope explicitly',
    }
    json.dump(stats, open(os.path.join(args.out_dir, 'candidate_stats.json'), 'w'), indent=1, ensure_ascii=False)

    # ------------------------------------------------------------- HTML
    rows = []
    for i, c in enumerate(top):
        s = c['mining_signals']
        fr = c['annotated_frames'][len(c['annotated_frames']) // 2]
        img = os.path.join(args.jpeg_root, c['video_id'], fr + '.jpg')
        rows.append(f"""<div class=card><div class=h>#{i+1} {c['query']}</div>
<img src="file://{img}" loading="lazy">
<div class=m>video={c['video_id']} exp={c['exp_id']} obj={c['target_obj_id']} cat={c['target_category']}<br>
same_cat={s['n_same_cat']} co_visible={s['co_visible_frames']} ({s['co_visible_ratio']:.2f}) min_dist={s['min_centroid_dist_norm']:.3f}<br>
max_dil_overlap={s['max_dilated_overlap']:.3f} gap={s['target_gap_frames']} absent={s['target_absent_frames']} qtype={s['query_type']} score={s['rank_score']:.2f}</div></div>""")
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Ambiguous referent candidates (top {len(top)})</title>
<style>body{{font-family:sans-serif;margin:20px}}.card{{display:inline-block;width:420px;margin:8px;border:1px solid #ccc;border-radius:8px;padding:8px;vertical-align:top}}
img{{width:100%;border-radius:4px}}.h{{font-weight:700;margin-bottom:4px}}.m{{font-size:12px;color:#333;margin-top:4px}}</style></head><body>
<h2>M5.1 ambiguous-referent candidate pool — top {len(top)} of {len(pool)}</h2>
<p>Real Ref-YT-VOS valid expressions + GT instance tracks only (no toy negatives). Ranking = z-score composite over mining signals.</p>
{''.join(rows)}</body></html>"""
    with open(os.path.join(args.out_dir, 'candidates.html'), 'w') as fh:
        fh.write(html)

    print(f'[mine] mined={len(cases)} pool={len(pool)} top_k={len(top)} -> {args.out_dir}', flush=True)
    print(f'[mine] query_type dist (pool): {stats["query_type_distribution"]}', flush=True)
    print(f'[mine] co_visible_ratio (all mined) mean={stats["mined_co_visible_ratio"]["mean"]:.3f} '
          f'max={stats["mined_co_visible_ratio"]["max"]:.3f}', flush=True)


if __name__ == '__main__':
    main()

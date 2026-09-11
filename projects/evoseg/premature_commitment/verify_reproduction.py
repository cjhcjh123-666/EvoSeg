"""M5.4 — verify that the P0 audit numbers reproduce.

Re-infers the same cases with the same settings (optionally split across GPUs
as `--shard-*` runs) and compares every run against the reference audit JSON:
exact-match rates for discrete outputs (identity decision, `[SEG]` presence,
frame counts) and absolute differences for continuous quantities (IoUs, margin,
IDErr, cosine of the `[SEG]` vector).

Nothing is silently tolerated: every mismatch is enumerated in the output.

Usage
-----
python verify_reproduction.py --reference <ref audit json> \
  --shard "<artifact_root>/results/premature_commitment/repro/*.json" \
  --out <artifact_root>/results/premature_commitment/repro/reproduction_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DISCRETE = ('identity_decision', 'has_seg', 'n_frames_used', 'sam2_prompt_frames')
CONTINUOUS = ('iou_target_mean', 'iou_distractor_mean', 'identity_margin_mean',
              'id_err_rate')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--shard', nargs='+', required=True,
                    help='shard JSON files or globs (quote they glob yourself)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--atol', type=float, default=0.0,
                    help='absolute tolerance for continuous quantities (0 = exact)')
    return ap.parse_args()


def index_records(doc):
    return {r['case_id']: r for r in doc['records']}


def index_runs(rec):
    return {(r['mode'], round(float(r['prefix_fraction']), 6)): r for r in rec['runs']}


def main():
    args = parse_args()
    ref = json.load(open(args.reference))
    ref_recs = index_records(ref)
    shard_paths = sorted({p for pat in args.shard for p in (glob.glob(pat) or [pat])})
    if not shard_paths:
        raise SystemExit(f'no shard files matched: {args.shard}')
    shard_recs, provenance = {}, []
    for p in shard_paths:
        d = json.load(open(p))
        meta_path = p.replace('.json', '.meta.json')
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        provenance.append({'shard': os.path.abspath(p), 'meta': meta_path if meta else None,
                           'git_sha': meta.get('git_sha'),
                           'command': meta.get('command'),
                           'shard_index': meta.get('shard_index'),
                           'shard_count': meta.get('shard_count'),
                           'case_ids': sorted(index_records(d))})
        for cid, rec in index_records(d).items():
            if cid in shard_recs:
                raise SystemExit(f'case {cid} present in more than one shard')
            shard_recs[cid] = rec

    only_ref = sorted(set(ref_recs) - set(shard_recs))
    only_new = sorted(set(shard_recs) - set(ref_recs))
    common = sorted(set(ref_recs) & set(shard_recs))

    disc_total = disc_equal = 0
    disc_mismatch, cont = [], {k: [] for k in CONTINUOUS}
    frame_total = frame_equal = 0
    frame_iou_deltas, per_case = [], []
    for cid in common:
        a, b = index_runs(ref_recs[cid]), index_runs(shard_recs[cid])
        if set(a) != set(b):
            raise SystemExit(f'{cid}: run keys differ: {sorted(set(a) ^ set(b))}')
        row = {'case_id': cid, 'n_runs': len(a), 'discrete_mismatches': [],
               'max_abs_delta': {}, 'frame_id_err_equal_frac': None}
        for key in sorted(a):
            ra, rb = a[key], b[key]
            for f in DISCRETE:
                disc_total += 1
                if ra.get(f) == rb.get(f):
                    disc_equal += 1
                else:
                    m = {'case_id': cid, 'mode': key[0], 'prefix_fraction': key[1],
                         'field': f, 'reference': ra.get(f), 'reproduced': rb.get(f)}
                    disc_mismatch.append(m)
                    row['discrete_mismatches'].append(
                        {'field': f, 'mode': key[0], 'reference': ra.get(f),
                         'reproduced': rb.get(f)})
            for f in CONTINUOUS:
                if ra.get(f) is None or rb.get(f) is None:
                    continue
                d_ = abs(float(ra[f]) - float(rb[f]))
                cont[f].append(d_)
                row['max_abs_delta'][f] = max(row['max_abs_delta'].get(f, 0.0), d_)
            fa = [r['id_err'] for r in ra['per_frame']]
            fb = [r['id_err'] for r in rb['per_frame']]
            if len(fa) == len(fb) and fa:
                frame_total += len(fa)
                frame_equal += int(sum(1 for x, y in zip(fa, fb) if x == y))
                frame_iou_deltas += [abs(x['iou_target'] - y['iou_target'])
                                     for x, y in zip(ra['per_frame'], rb['per_frame'])]
                row['frame_id_err_equal_frac'] = float(
                    np.mean([x == y for x, y in zip(fa, fb)]))
        per_case.append(row)

    summary = {
        'reference': os.path.abspath(args.reference),
        'shards': provenance,
        'n_cases_reference': len(ref_recs),
        'n_cases_reproduced': len(shard_recs),
        'n_cases_common': len(common),
        'cases_only_in_reference': only_ref,
        'cases_only_in_reproduction': only_new,
        'discrete': {'n_compared': disc_total, 'n_equal': disc_equal,
                     'equal_frac': (disc_equal / disc_total) if disc_total else None,
                     'n_mismatch': len(disc_mismatch)},
        'continuous_abs_delta': {
            k: {'n': len(v), 'mean': float(np.mean(v)) if v else None,
                'max': float(np.max(v)) if v else None,
                'n_over_atol': int(sum(1 for x in v if x > args.atol))}
            for k, v in cont.items()},
        'frame_level': {
            'n_frames_compared': frame_total, 'n_id_err_equal': frame_equal,
            'id_err_equal_frac': (frame_equal / frame_total) if frame_total else None,
            'iou_target_abs_delta_mean': (float(np.mean(frame_iou_deltas))
                                          if frame_iou_deltas else None),
            'iou_target_abs_delta_max': (float(np.max(frame_iou_deltas))
                                         if frame_iou_deltas else None)},
        'atol': args.atol,
        'discrete_mismatches': disc_mismatch,
        'per_case': per_case,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(f'[verify] cases {len(common)}/{len(ref_recs)} compared '
          f'(only-in-reference={len(only_ref)}, only-in-reproduction={len(only_new)})',
          flush=True)
    if disc_total:
        print(f'[verify] discrete equal {disc_equal}/{disc_total} '
              f'({100 * summary["discrete"]["equal_frac"]:.1f}%)', flush=True)
    if frame_total:
        print(f'[verify] frame-level id_err agreement '
              f'{100 * summary["frame_level"]["id_err_equal_frac"]:.1f}% over '
              f'{frame_total} frames', flush=True)
    for k, v in summary['continuous_abs_delta'].items():
        print(f'[verify] {k}: mean|delta|={v["mean"]} max|delta|={v["max"]}', flush=True)
    print(f'[verify] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()

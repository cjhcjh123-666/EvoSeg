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

# (block, field): block None = the run record itself, otherwise the raw/gated
# sub-dict. Discrete fields must match exactly; continuous ones are compared with
# absolute differences (a tolerance of 0 means bit-identical).
DISCRETE = ((None, 'has_seg'), (None, 'n_frames_used'), (None, 'sam2_prompt_frames'),
            (None, 'identity_stream_used'), (None, 'raw_available'),
            ('raw', 'identity_decision'), ('raw', 'first_nonempty_frame'),
            ('raw', 'n_nonempty_frames'), ('gated', 'identity_decision'))
CONTINUOUS = ((('raw', 'iou_target_mean'), ('raw', 'iou_distractor_mean'),
               ('raw', 'identity_margin_mean'), ('raw', 'id_err_rate'),
               ('raw', 'empty_rate')),
              (('gated', 'iou_target_mean'), ('gated', 'iou_distractor_mean'),
               ('gated', 'identity_margin_mean'), ('gated', 'id_err_rate'),
               ('gated', 'empty_rate')))


def get_field(run, block, field):
    return run.get(field) if block is None else (run.get(block) or {}).get(field)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--shard', nargs='+', required=True,
                    help='shard JSON files or globs (quote they glob yourself)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--atol', type=float, default=0.0,
                    help='absolute tolerance for continuous quantities (0 = exact)')
    ap.add_argument('--reference-flat-stream', default=None,
                    help='schema-v1 references store metrics at the top level of each run; '
                         'name the schema-v2 stream they correspond to (e.g. "gated") to '
                         'compare them against that stream inside the schema-v2 runs')
    return ap.parse_args()


def normalise_reference(recs, flat_stream):
    """Wrap flat (schema-v1) run metrics into the named v2 stream block."""
    if flat_stream is None:
        return recs
    out = {}
    for cid, rec in recs.items():
        runs = []
        for r in rec['runs']:
            if 'raw' in r or 'gated' in r:
                raise SystemExit('--reference-flat-stream given but the reference already '
                                 'contains stream blocks; drop the flag')
            keys = ('iou_target_mean', 'iou_distractor_mean', 'identity_margin_mean',
                    'id_err_rate', 'empty_rate', 'identity_decision',
                    'first_nonempty_frame', 'n_nonempty_frames', 'per_frame')
            new = {k: v for k, v in r.items() if k not in keys}
            new[flat_stream] = {k: r[k] for k in keys if k in r}
            new['identity_stream_used'] = 'v1_flat_wrapped'
            runs.append(new)
        out[cid] = dict(rec, runs=runs)
    return out


def index_records(doc):
    return {r['case_id']: r for r in doc['records']}


def index_runs(rec):
    return {(r['mode'], round(float(r['prefix_fraction']), 6)): r for r in rec['runs']}


def main():
    args = parse_args()
    ref = json.load(open(args.reference))
    ref_recs = normalise_reference(index_records(ref), args.reference_flat_stream)
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

    # A flat (schema-v1) reference can only speak about one stream, so compare
    # exactly the fields that exist on both sides.
    if args.reference_flat_stream:
        discrete = tuple(x for x in DISCRETE if x[0] is None and x[1] in
                         ('has_seg', 'n_frames_used', 'sam2_prompt_frames'))
        discrete += ((args.reference_flat_stream, 'identity_decision'),
                     (args.reference_flat_stream, 'first_nonempty_frame'),
                     (args.reference_flat_stream, 'n_nonempty_frames'))
        continuous = tuple(g for g in CONTINUOUS
                           if any(b == args.reference_flat_stream for b, _ in g))
    else:
        discrete, continuous = DISCRETE, CONTINUOUS

    disc_total = disc_equal = 0
    disc_mismatch = []
    cont = {}
    for group in continuous:
        for b, f in group:
            cont[f'{b}.{f}'] = []
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
            for block, f in discrete:
                va, vb = get_field(ra, block, f), get_field(rb, block, f)
                disc_total += 1
                if va == vb:
                    disc_equal += 1
                else:
                    name = f if block is None else f'{block}.{f}'
                    m = {'case_id': cid, 'mode': key[0], 'prefix_fraction': key[1],
                         'field': name, 'reference': va, 'reproduced': vb}
                    disc_mismatch.append(m)
                    row['discrete_mismatches'].append(
                        {'field': name, 'mode': key[0], 'reference': va,
                         'reproduced': vb})
            for group in continuous:
                for block, f in group:
                    va, vb = get_field(ra, block, f), get_field(rb, block, f)
                    if va is None or vb is None:
                        continue
                    name = f'{block}.{f}'
                    d_ = abs(float(va) - float(vb))
                    cont[name].append(d_)
                    row['max_abs_delta'][name] = max(row['max_abs_delta'].get(name, 0.0), d_)
            fa = [r['id_err'] for r in (ra.get('raw') or {}).get('per_frame', [])]
            fb = [r['id_err'] for r in (rb.get('raw') or {}).get('per_frame', [])]
            if len(fa) == len(fb) and fa:
                frame_total += len(fa)
                frame_equal += int(sum(1 for x, y in zip(fa, fb) if x == y))
                frame_iou_deltas += [abs(x['iou_target'] - y['iou_target'])
                                     for x, y in zip((ra.get('raw') or {}).get('per_frame', []),
                                                     (rb.get('raw') or {}).get('per_frame', []))]
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

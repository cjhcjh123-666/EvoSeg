"""M5.3 (v2) — delayed / ORACLE_* headroom, horizon-matched.

Why there is no fair `DELAYED` variant
--------------------------------------
`predict_forward` produces exactly one `[SEG]` per query, applies its language
prompt on the first `min(5, T)` frames and runs a single causal SAM2
propagation. There is no second commitment point, no way to hold a hypothesis
and re-decide later, and no state a delayed initialisation could preserve.
Truncating the input (the prefix experiment) is *not* a delayed-commit
mechanism: the model still commits inside whatever prefix it is given.

The v1 version of this script compared each prefix over its **own** horizon,
which made the "oracle" trivially prefer the shortest prefix: a 5-frame horizon
is easier than a 24-frame horizon (later frames are where occlusion, crossing
and identity switches live). That confound is removed here:

  * `ORACLE_TIMING` and `NORMAL` are both computed on the strict per-case common
    window (frames 0..W_i-1, W_i = min sam2_prompt_frames over the compared
    prefixes), so every prefix sees the same frames;
  * the old full-horizon variant is kept as `ORACLE_TIMING_FULL_HORIZON` and
    explicitly labelled confounded, so the size of the confound is visible.

Variants
--------
  NORMAL                     the official full-video path, on the common window
  ORACLE_TIMING              per-case best prefix chosen with GT, common window
  ORACLE_TIMING_FULL_HORIZON per-case best prefix chosen with GT, own horizons
                             (CONFOUNDED — reported for transparency only)
  DELAYED                    not implementable (documented, not hacked)
  ORACLE_ID                  not implementable as a method (1 hypothesis)

Usage
-----
python eval_delayed_commit.py --audit "<artifact>/v2/*_shard[0-9].json" \
  --out-dir "<artifact>/headroom_v2"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_prefix_identity import (DEFAULT_MODE, PREFIX_MODE, agg,  # noqa: E402
                                  case_metrics, get_run, load_records)
from metrics import paired_delta  # noqa: E402
from utils import (DEFAULT_ANN_ROOT, DEFAULT_JPEG_ROOT, DEFAULT_MANIFEST,  # noqa: E402
                   DEFAULT_MODEL, git_sha, write_run_metadata)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--stream', default='raw', choices=['raw', 'gated'])
    ap.add_argument('--normal-fraction', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs, provenance = load_records(args.audit)
    fr = sorted({r['prefix_fraction'] for rec in recs.values()
                 for r in rec['runs'] if r['mode'] == PREFIX_MODE})

    def W_of(cid):
        lens = [r['sam2_prompt_frames'] for r in recs[cid]['runs']
                if r['mode'] == PREFIX_MODE]
        return int(min([args.window] + lens)) if lens else args.window

    normal, normal_full = [], []
    oracle_cw, oracle_fh = [], []
    hist_cw, hist_fh = {f: 0 for f in fr}, {f: 0 for f in fr}
    rows = []
    for cid, rec in recs.items():
        W = W_of(cid)
        runs = {f: get_run(rec, PREFIX_MODE, f) for f in fr}
        runs = {f: r for f, r in runs.items() if r}
        if not runs:
            continue
        m_norm = case_metrics(runs[args.normal_fraction], args.stream, W)
        m_norm_fh = case_metrics(runs[args.normal_fraction], args.stream, None)
        per_prefix = {f: case_metrics(r, args.stream, W) for f, r in runs.items()}
        per_prefix = {f: m for f, m in per_prefix.items() if m}
        if m_norm is None or not per_prefix:
            continue
        best_cw = min(per_prefix, key=lambda f: per_prefix[f]['id_err'])
        best_fh = min(runs, key=lambda f: case_metrics(runs[f], args.stream, None)['id_err'])
        normal.append(m_norm['id_err']); normal_full.append(m_norm_fh['id_err'])
        oracle_cw.append(per_prefix[best_cw]['id_err'])
        oracle_fh.append(case_metrics(runs[best_fh], args.stream, None)['id_err'])
        hist_cw[best_cw] += 1
        hist_fh[best_fh] += 1
        rows.append({'case_id': cid, 'window_frames': W,
                     'id_err_normal': m_norm['id_err'],
                     'id_err_normal_full_horizon': m_norm_fh['id_err'],
                     'best_prefix_common_window': best_cw,
                     'id_err_oracle_common_window': per_prefix[best_cw]['id_err'],
                     'best_prefix_full_horizon': best_fh,
                     'id_err_oracle_full_horizon': case_metrics(runs[best_fh],
                                                                args.stream, None)['id_err']})

    out = {
        'schema_version': 2,
        'n_cases': len(rows), 'prefixes': fr, 'stream': args.stream,
        'audit_files': provenance,
        'normal_fraction': args.normal_fraction,
        'window_rule': 'W_i = min(sam2_prompt_frames over prefixes), capped at '
                       f'{args.window}; identical frames for every prefix',
        'NORMAL': {'id_err': agg(normal, args.bootstrap, args.seed)},
        'ORACLE_TIMING': {
            'note': 'GT used only to choose the prefix per case; upper bound, never a method',
            'window': 'common window (horizon-matched)',
            'id_err': agg(oracle_cw, args.bootstrap, args.seed),
            'paired_delta_vs_normal': paired_delta(normal, oracle_cw, args.seed,
                                                   args.bootstrap),
            'best_prefix_histogram': hist_cw},
        'ORACLE_TIMING_FULL_HORIZON': {
            'note': 'CONFOUNDED: each prefix scored over its own horizon, so the shortest '
                    'prefix is favoured by construction. This is what the v1 report quoted; '
                    'do not use it as evidence about evidence volume.',
            'window': 'per-prefix horizon (not comparable)',
            'id_err': agg(oracle_fh, args.bootstrap, args.seed),
            'paired_delta_vs_normal_full_horizon': paired_delta(normal_full, oracle_fh,
                                                                args.seed, args.bootstrap),
            'best_prefix_histogram': hist_fh},
        'DELAYED': {
            'implementable': False,
            'reason': 'single [SEG] -> single prompt on first min(5,T) frames -> single causal '
                      'propagation; no second commitment point and no hypothesis state'},
        'ORACLE_ID': {
            'implementable_as_method': False,
            'reason': 'the model holds exactly one referent hypothesis; an ID oracle would '
                      'have to inject GT masks, which is not a method',
            'n_hypotheses_available': 1},
        'per_case': rows,
    }
    json.dump(out, open(os.path.join(args.out_dir, 'delayed_oracle_headroom_v2.json'), 'w'),
              ensure_ascii=False)
    write_run_metadata(
        os.path.join(args.out_dir, 'delayed_oracle_headroom_v2.meta.json'),
        command=' '.join(sys.argv), script=os.path.abspath(__file__),
        git_sha=git_sha(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        audit_files=[p['path'] for p in provenance],
        model=DEFAULT_MODEL, dataset='Ref-YT-VOS valid',
        ann_root=DEFAULT_ANN_ROOT, jpeg_root=DEFAULT_JPEG_ROOT,
        manifest=DEFAULT_MANIFEST, seed=args.seed,
        bootstrap_resamples=args.bootstrap, stream=args.stream,
        window_cap=args.window, normal_fraction=args.normal_fraction,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'none'), world_size=1,
        new_inference=False,
        oracle_usage='ORACLE_TIMING* uses GT only to choose the prefix (upper bound, not a '
                     'method); DELAYED/ORACLE_ID are non-implementable, not methods',
        n_cases=len(rows))
    print(f'[headroom v2] stream={args.stream} NORMAL={np.mean(normal) * 100:.1f}% '
          f'ORACLE_TIMING(common window)={np.mean(oracle_cw) * 100:.1f}% '
          f'hist={hist_cw}', flush=True)
    print(f'[headroom v2] (confounded, for comparison) ORACLE full-horizon='
          f'{np.mean(oracle_fh) * 100:.1f}% hist={hist_fh}', flush=True)


if __name__ == '__main__':
    main()

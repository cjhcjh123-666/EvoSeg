"""M5.3 — delayed / oracle headroom diagnostics (architecture-neutral).

Why there is no fair `DELAYED` variant
--------------------------------------
`modeling_sa2va_qwen.predict_forward` produces exactly one `[SEG]` per query,
applies its language prompt on the first `min(5, T)` frames and then runs a single
causal SAM2 propagation (see P0 audit §1). There is no second commitment point,
no way to "hold" a hypothesis and re-decide later, and no state that a delayed
initialisation could preserve. Truncating the input (the prefix experiment) is
the only lever the architecture exposes, and it is *not* a delayed-commit
mechanism: the model still commits inside whatever prefix it is given.

Rather than hacking a variant that only looks better, we report the
architecture-neutral diagnostics that answer the same scientific question:

  NORMAL            prefix 1.0 (official full-video path)
  DELAYED           not implementable -> documented, see above
  ORACLE_TIMING     per-case best prefix chosen with GT (upper bound on "if we
                    knew when to commit"); labelled ORACLE everywhere
  ORACLE_ID         degenerate: the model holds exactly ONE hypothesis, so there
                    is no alternative to select; any ID oracle would have to
                    inject GT masks and is therefore not a method

Usage
-----
python eval_delayed_commit.py --audit <...>/P0_prefix_audit_identity_swap_n20.json \
  --out-dir <...>/premature_commitment/headroom
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import bootstrap_ci, paired_delta  # noqa: E402
from utils import (DEFAULT_ANN_ROOT, DEFAULT_JPEG_ROOT, DEFAULT_MANIFEST,  # noqa: E402
                   DEFAULT_MODEL, git_sha, write_run_metadata)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix-mode', default='prefix_vlm_all_frames')
    ap.add_argument('--normal-fraction', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def get_run(case, mode, frac=None):
    for r in case['runs']:
        if r['mode'] == mode and (frac is None or abs(r['prefix_fraction'] - frac) < 1e-9):
            return r
    return None


def win_margin(run, W):
    pf = [r for r in run['per_frame'] if r['t'] < W]
    if not pf:
        return None
    return float(np.mean([r['identity_margin'] for r in pf]))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = json.load(open(args.audit))
    recs = d['records']
    fr = sorted({r['prefix_fraction'] for c in recs for r in c['runs'] if r['mode'] == args.prefix_mode})
    W = args.window

    normal, oracle_full, rows = [], [], []
    normal_win, best_win = [], []
    short_prefix_better = 0
    for c in recs:
        runs = {f: get_run(c, args.prefix_mode, f) for f in fr}
        runs = {f: r for f, r in runs.items() if r}
        if not runs:
            continue
        normal_err = runs[args.normal_fraction]['id_err_rate']
        # ORACLE_TIMING (full horizon): pick the prefix with the lowest IDErr using GT
        best_f = min(runs, key=lambda f: runs[f]['id_err_rate'])
        oracle_err = runs[best_f]['id_err_rate']
        # ORACLE_TIMING (identical window): pick the prefix with the best early margin
        margins = {f: win_margin(r, W) for f, r in runs.items()}
        margins = {f: m for f, m in margins.items() if m is not None}
        best_w = max(margins, key=lambda f: margins[f]) if margins else None
        normal_win_margin = margins.get(args.normal_fraction)
        best_win_margin = margins[best_w] if best_w else None
        normal.append(normal_err); oracle_full.append(oracle_err)
        if normal_win_margin is not None and best_win_margin is not None:
            normal_win.append(normal_win_margin); best_win.append(best_win_margin)
        if best_w is not None and best_w < max(fr):
            short_prefix_better += 1
        rows.append({'case_id': c['case_id'],
                     'id_err_normal': normal_err,
                     'oracle_timing_best_prefix': best_f,
                     'id_err_oracle_timing': oracle_err,
                     'normal_window_margin': normal_win_margin,
                     'best_window_margin': best_win_margin})
    n = len(rows)
    out = {
        'n_cases': n,
        'prefixes': fr,
        'normal_fraction': args.normal_fraction,
        'NORMAL': {'id_err': bootstrap_ci(normal, args.bootstrap, args.seed)},
        'ORACLE_TIMING': {
            'note': 'per-case best prefix chosen with GT; upper bound, never a method',
            'id_err': bootstrap_ci(oracle_full, args.bootstrap, args.seed),
            'paired_delta_vs_normal': paired_delta(normal, oracle_full, args.seed, args.bootstrap),
            'best_prefix_histogram': {str(f): sum(1 for r in rows
                                                  if r['oracle_timing_best_prefix'] == f)
                                      for f in fr},
        },
        'ORACLE_TIMING_SAME_WINDOW': {
            'note': 'same first-%d-frame window for every prefix; isolates the effect of the extra '
                    'VLM evidence on the *identical* frames, excluding propagation horizon' % W,
            'window_frames': W,
            'normal_margin': bootstrap_ci(normal_win, args.bootstrap, args.seed),
            'best_margin': bootstrap_ci(best_win, args.bootstrap, args.seed),
            'paired_delta_best_minus_normal': paired_delta(normal_win, best_win, args.seed,
                                                           args.bootstrap),
            'cases_where_best_prefix_is_shorter_than_full': short_prefix_better,
        },
        'DELAYED': {
            'implementable': False,
            'reason': 'single [SEG] -> single prompt on first min(5,T) frames -> single causal '
                      'propagation; no second commitment point and no hypothesis state to delay',
            'architecture_neutral_analogue': 'prefix truncation (reported in prefix_identity_eval.json)',
        },
        'ORACLE_ID': {
            'implementable_as_method': False,
            'reason': 'the model holds exactly one referent hypothesis; an ID oracle would have to '
                      'inject GT masks, which is not a method',
            'n_hypotheses_available': 1,
        },
        'per_case': rows,
    }
    json.dump(out, open(os.path.join(args.out_dir, 'delayed_oracle_headroom.json'), 'w'),
              ensure_ascii=False)
    write_run_metadata(
        os.path.join(args.out_dir, 'delayed_oracle_headroom.meta.json'),
        command=' '.join(sys.argv),
        script=os.path.abspath(__file__),
        git_sha=git_sha(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        audit_json=os.path.abspath(args.audit),
        model=DEFAULT_MODEL,
        dataset='Ref-YT-VOS valid',
        ann_root=DEFAULT_ANN_ROOT, jpeg_root=DEFAULT_JPEG_ROOT,
        manifest=DEFAULT_MANIFEST,
        seed=args.seed, bootstrap_resamples=args.bootstrap,
        prefix_mode=args.prefix_mode, normal_fraction=args.normal_fraction,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'none'),
        world_size=1,
        vlm_all_frames=True,
        new_inference=False,
        oracle_usage='ORACLE_TIMING* uses GT only to choose the prefix (upper bound, not a method); '
                     'DELAYED/ORACLE_ID are reported as not implementable, never as methods',
        note='consumes the P0 audit JSON; no model is loaded and no GPU is used',
        n_cases=out['n_cases'], prefixes=fr,
    )
    nm = float(np.mean(normal)) if normal else float('nan')
    om = float(np.mean(oracle_full)) if oracle_full else float('nan')
    nwm = float(np.mean(normal_win)) if normal_win else float('nan')
    bwm = float(np.mean(best_win)) if best_win else float('nan')
    print(f'[headroom] NORMAL IDErr={nm*100:.1f}%  ORACLE_TIMING IDErr={om*100:.1f}% '
          f'(delta {100*(om-nm):+.1f}pp, ORACLE)', flush=True)
    print(f'[headroom] same-window margin NORMAL={nwm:+.3f} vs ORACLE best={bwm:+.3f}; '
          f'best prefix shorter than full in {short_prefix_better}/{n} cases', flush=True)


if __name__ == '__main__':
    main()

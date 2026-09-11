"""M5.3 — prefix identity curve, full-video vs prefix, error persistence.

Consumes the P0 audit JSON (no new inference) and produces:

  * prefix identity curve (case-macro + frame-micro) with bootstrap CIs;
  * a paired comparison on IDENTICAL early frames across evidence volumes
    (prefix 0.2 vs prefix 1.0) — this isolates the effect of extra evidence
    from propagation-horizon effects;
  * error-correction table (does more evidence fix a wrong referent?);
  * a plot for the pilot report.

Usage
-----
python eval_prefix_identity.py \
  --audit <artifact_root>/results/premature_commitment/P0_prefix_audit_identity_swap_n20.json \
  --out-dir <artifact_root>/results/premature_commitment/prefix_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (DEFAULT_ANN_ROOT, DEFAULT_JPEG_ROOT, DEFAULT_MANIFEST,  # noqa: E402
                   DEFAULT_MODEL, git_sha, write_run_metadata)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as exc:  # recorded, not hidden
    print(f'[prefix-eval] matplotlib unavailable: {exc}', flush=True)
    HAVE_MPL = False

from metrics import bootstrap_ci, paired_delta  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix-mode', default='prefix_vlm_all_frames')
    ap.add_argument('--default-mode', default='full_video_default_vlm_first5')
    ap.add_argument('--window', type=int, default=5,
                    help='frames used for paired same-window comparisons')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def get_run(case, mode, frac=None):
    for r in case['runs']:
        if r['mode'] == mode and (frac is None or abs(r['prefix_fraction'] - frac) < 1e-9):
            return r
    return None


def window_metrics(run, a, b):
    pf = [r for r in run['per_frame'] if a <= r['t'] < b]
    if not pf:
        return None
    it = float(np.mean([r['iou_target'] for r in pf]))
    idd = float(np.mean([r['iou_distractor_max'] for r in pf]))
    return {'n': len(pf), 'iou_target': it, 'iou_distractor': idd,
            'margin': it - idd, 'id_err': float(np.mean([r['id_err'] for r in pf])),
            'empty': float(np.mean([r['empty'] for r in pf]))}


def micro_from_frames(pf):
    if not pf:
        return {}
    it = float(np.mean([r['iou_target'] for r in pf]))
    idd = float(np.mean([r['iou_distractor_max'] for r in pf]))
    return {'n_frames': len(pf), 'iou_target': it, 'iou_distractor': idd,
            'margin': it - idd,
            'id_err': float(np.mean([r['id_err'] for r in pf])),
            'empty_rate': float(np.mean([r['empty'] for r in pf]))}


def frame_micro(runs):
    """Pool every frame of every run: frame-micro view of the same quantity."""
    return micro_from_frames([r for run in runs for r in run['per_frame']])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = json.load(open(args.audit))
    recs = d['records']
    fr = sorted({r['prefix_fraction'] for c in recs for r in c['runs'] if r['mode'] == args.prefix_mode})
    out = {'n_cases': len(recs), 'prefixes': fr, 'window_frames': args.window}

    # ---------------- prefix curve ----------------
    curve = []
    for f in fr:
        runs = [get_run(c, args.prefix_mode, f) for c in recs]
        runs = [r for r in runs if r]
        row = {'prefix_fraction': f,
               'mean_frames': float(np.mean([r['n_frames_used'] for r in runs])),
               'iou_target': bootstrap_ci([r['iou_target_mean'] for r in runs], args.bootstrap, args.seed),
               'iou_distractor': bootstrap_ci([r['iou_distractor_mean'] for r in runs], args.bootstrap, args.seed),
               'margin': bootstrap_ci([r['identity_margin_mean'] for r in runs], args.bootstrap, args.seed),
               'id_err': bootstrap_ci([r['id_err_rate'] for r in runs], args.bootstrap, args.seed),
               'decisions': {k: sum(1 for r in runs if r['identity_decision'] == k)
                             for k in ('target', 'distractor', 'empty')},
               'cos_to_full': float(np.mean([r['seg_vec_cos_to_full'] for r in runs
                                             if r['seg_vec_cos_to_full'] is not None])),
               'frame_micro': frame_micro(runs),
               }
        curve.append(row)
    out['prefix_curve'] = curve

    # ---------------- default path ----------------
    druns = [get_run(c, args.default_mode) for c in recs]
    druns = [r for r in druns if r]
    out['default_path'] = {
        'iou_target': float(np.mean([r['iou_target_mean'] for r in druns])),
        'iou_distractor': float(np.mean([r['iou_distractor_mean'] for r in druns])),
        'id_err': float(np.mean([r['id_err_rate'] for r in druns])),
        'decisions': {k: sum(1 for r in druns if r['identity_decision'] == k)
                      for k in ('target', 'distractor', 'empty')},
        'frame_micro': frame_micro(druns),
    }

    # ---------------- paired, identical window ----------------
    W = args.window
    lo_frac, hi_frac = min(fr), max(fr)
    e_hi, e_lo, flips, agree = [], [], 0, 0
    corr = {'wrong_fixed': 0, 'wrong_persist': 0, 'right_right': 0, 'right_broken': 0}
    frames_lo, frames_hi = [], []
    rows = []
    for c in recs:
        r_lo = get_run(c, args.prefix_mode, lo_frac)
        r_hi = get_run(c, args.prefix_mode, hi_frac)
        if not r_lo or not r_hi:
            continue
        w_lo = window_metrics(r_lo, 0, W)
        w_hi = window_metrics(r_hi, 0, W)
        if not w_lo or not w_hi:
            continue
        e_lo.append(w_lo['id_err']); e_hi.append(w_hi['id_err'])
        frames_lo += [r for r in r_lo['per_frame'] if r['t'] < W]
        frames_hi += [r for r in r_hi['per_frame'] if r['t'] < W]
        wrong_lo = w_lo['margin'] < 0
        wrong_hi = w_hi['margin'] < 0
        if wrong_lo and not wrong_hi:
            corr['wrong_fixed'] += 1
        elif wrong_lo and wrong_hi:
            corr['wrong_persist'] += 1
        elif not wrong_lo and not wrong_hi:
            corr['right_right'] += 1
        else:
            corr['right_broken'] += 1
        if r_lo['identity_decision'] != r_hi['identity_decision']:
            flips += 1
        rows.append({'case_id': c['case_id'], 'query': c['query'],
                     'id_err_low_evidence': w_lo['id_err'], 'id_err_full_evidence': w_hi['id_err'],
                     'margin_low_evidence': w_lo['margin'], 'margin_full_evidence': w_hi['margin'],
                     'cos_seg_to_full': r_lo.get('seg_vec_cos_to_full')})
    n = len(rows)
    out['paired_same_window'] = {
        'window_frames': W, 'n_cases': n,
        'id_err_low_evidence': bootstrap_ci(e_lo, args.bootstrap, args.seed),
        'id_err_full_evidence': bootstrap_ci(e_hi, args.bootstrap, args.seed),
        'delta_id_err_full_minus_low': paired_delta(e_lo, e_hi, args.seed, args.bootstrap),
        'frame_micro': {'low_evidence': micro_from_frames(frames_lo),
                        'full_evidence': micro_from_frames(frames_hi)},
        'error_transitions': corr,
        'decision_flip_cases': flips,
        'cos_seg_low_to_full_mean': float(np.mean([r['cos_seg_to_full'] for r in rows
                                                   if r['cos_seg_to_full'] is not None])),
        'per_case': rows,
    }
    out['caveats'] = [
        'prefix runs share the same first min(5,T) SAM2 prompt frames, so the paired '
        'same-window comparison isolates the effect of extra VLM evidence from the '
        'propagation horizon.',
        'these are development/feasibility cases (no human identifiability labels yet).',
    ]

    with open(os.path.join(args.out_dir, 'prefix_identity_eval.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False)
    write_run_metadata(
        os.path.join(args.out_dir, 'prefix_identity_eval.meta.json'),
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
        prefix_mode=args.prefix_mode, default_mode=args.default_mode,
        window_frames=args.window,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'none'),
        world_size=1,
        vlm_all_frames=True,
        new_inference=False,
        note='consumes the P0 audit JSON; no model is loaded and no GPU is used',
        n_cases=out['n_cases'], prefixes=out['prefixes'],
    )

    # ---------------- plot ----------------
    if HAVE_MPL:
        xs = [r['mean_frames'] for r in curve]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].errorbar(xs, [r['id_err']['mean'] for r in curve],
                       yerr=[[r['id_err']['mean'] - r['id_err']['lo'] for r in curve],
                             [r['id_err']['hi'] - r['id_err']['mean'] for r in curve]],
                       marker='o', label='IDErr (prefix)')
        ax[0].plot(xs, [r['iou_target']['mean'] for r in curve], marker='s', label='IoU target')
        ax[0].plot(xs, [r['iou_distractor']['mean'] for r in curve], marker='^', label='IoU distractor')
        ax[0].set_xlabel('frames of evidence given to the VLM')
        ax[0].set_title('identity vs evidence volume (case-macro, 20 cases)')
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(xs, [r['cos_to_full'] for r in curve], marker='o')
        ax[1].set_ylim(0.9, 1.01); ax[1].set_xlabel('frames of evidence')
        ax[1].set_title('cos([SEG], [SEG]@full video)'); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'prefix_identity_curve.png'), dpi=160)
        print(f'[prefix-eval] plot -> {args.out_dir}/prefix_identity_curve.png', flush=True)

    print(f'[prefix-eval] cases={n} -> {args.out_dir}/prefix_identity_eval.json', flush=True)
    print(f'[prefix-eval] IDErr low-evidence={np.mean(e_lo)*100:.1f}% full-evidence={np.mean(e_hi)*100:.1f}% '
          f'| corrections={corr["wrong_fixed"]}/{corr["wrong_fixed"]+corr["wrong_persist"]}', flush=True)


if __name__ == '__main__':
    main()

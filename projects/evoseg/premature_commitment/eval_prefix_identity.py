"""M5.3 (v2) — prefix identity analysis on RAW masks with a strict common window.

Two corrections over the v1 evaluator, both required before any cross-prefix
comparison is reported:

1. **Raw masks.** Identity is computed on `raw_masks` (SAM2 propagation before
   the temporal verifier gate), not on `prediction_masks` (= raw * e_t). The
   gated stream mixes referent identity with the verifier's abstention, and an
   empty (gated-off) frame scores "no identity error" under the IoU comparison.
   Gated numbers are reported as a clearly labelled secondary stream.

2. **Strict per-case common window.** Prefixes have different horizons
   (prefix 0.2 sees ~5 frames, prefix 1.0 sees ~24), so comparing their own
   averages compares different time intervals — later frames are exactly where
   occlusion/crossing/identity switches live. Every cross-prefix number here is
   therefore computed on the intersection of the horizons:

       W_i = min over compared runs of `sam2_prompt_frames`  (and <= --window)

   and both sides use exactly the frame indices 0..W_i-1 of case i. Because
   `sam2_prompt_frames` = min(5, n_used), all frames in the window are prompted
   in every compared run, so within the window the runs differ only in the
   language conditioning produced by the VLM.

Also reports the cosine calibration obtained from the extra "different query,
same video" run, without which a cosine value between two [SEG] vectors has no
interpretable scale.

Usage
-----
python eval_prefix_identity.py --audit "<artifact>/v2/*_shard[0-9].json" \
  --out-dir "<artifact>/prefix_eval_v2"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import bootstrap_ci, paired_delta  # noqa: E402
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

PREFIX_MODE = 'prefix_vlm_all_frames'
DEFAULT_MODE = 'full_video_default_vlm_first5'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', nargs='+', required=True,
                    help='schema-v2 audit JSON file(s) or globs')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--window', type=int, default=5,
                    help='upper bound on the common window length')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--paired-lo', type=float, default=None,
                    help='low-evidence prefix for the paired test (default: min)')
    ap.add_argument('--paired-hi', type=float, default=None,
                    help='high-evidence prefix for the paired test (default: max)')
    return ap.parse_args()


def load_records(paths):
    recs, provenance = {}, []
    for path in sorted({p for pat in paths for p in (glob.glob(pat) or [pat])}):
        d = json.load(open(path))
        if d.get('schema_version') != 2:
            raise SystemExit(
                f'{path}: schema_version={d.get("schema_version")} — this evaluator '
                f'needs the v2 audit (raw_masks + gated). Re-run '
                f'audit_inference_information_flow.py first.')
        for r in d['records']:
            if r['case_id'] in recs:
                raise SystemExit(f'duplicate case {r["case_id"]} across audit files')
            recs[r['case_id']] = r
        provenance.append({'path': os.path.abspath(path), 'n_cases': len(d['records'])})
    return recs, provenance


def get_run(rec, mode, frac=None):
    for r in rec['runs']:
        if r['mode'] == mode and (frac is None or abs(r['prefix_fraction'] - frac) < 1e-9):
            return r
    return None


def agg(values, bootstrap=1000, seed=0):
    """case-macro aggregate with bootstrap CI over cases."""
    return bootstrap_ci(values, bootstrap, seed)


def micro(frames):
    """frame-micro aggregate over an explicit list of per-frame dicts."""
    if not frames:
        return {}
    it = float(np.mean([r['iou_target'] for r in frames]))
    idd = float(np.mean([r['iou_distractor_max'] for r in frames]))
    return {'n_frames': len(frames), 'iou_target': it, 'iou_distractor': idd,
            'margin': it - idd,
            'id_err': float(np.mean([r['id_err'] for r in frames])),
            'empty_rate': float(np.mean([r['empty'] for r in frames]))}


def stream_of(run, stream):
    return run.get(stream) if isinstance(run, dict) else None


def window_frames(run, stream, W):
    """Per-frame dicts of one run restricted to t < W (the common window)."""
    b = stream_of(run, stream)
    if not b:
        return None
    return [f for f in b['per_frame'] if f['t'] < W]


def case_metrics(run, stream, W=None):
    """Case-level metrics of one run, optionally restricted to t < W."""
    b = stream_of(run, stream)
    if not b:
        return None
    pf = b['per_frame'] if W is None else [f for f in b['per_frame'] if f['t'] < W]
    if not pf:
        return None
    nonempty = [f for f in pf if not f['empty']]
    it = float(np.mean([f['iou_target'] for f in pf]))
    idd = float(np.mean([f['iou_distractor_max'] for f in pf]))
    return {'n': len(pf), 'iou_target': it, 'iou_distractor': idd,
            'margin': it - idd,
            'id_err': float(np.mean([f['id_err'] for f in pf])),
            'empty_rate': float(np.mean([f['empty'] for f in pf])),
            'identity_decision': ('empty' if not nonempty else
                                  'target' if np.mean([f['identity_margin'] for f in nonempty]) >= 0
                                  else 'distractor')}


def curve(recs, fr, stream, W_of=None, bootstrap=1000, seed=0):
    """Per-prefix aggregates. W_of(case)->int restricts to a common window."""
    rows = []
    for f in fr:
        runs = [(cid, get_run(r, PREFIX_MODE, f)) for cid, r in recs.items()]
        runs = [(cid, r) for cid, r in runs if r]
        per = []
        frames = []
        n_window = []
        cos_vals = []
        for cid, r in runs:
            W = None if W_of is None else W_of(cid)
            m = case_metrics(r, stream, W)
            if m:
                per.append(m)
                pf = (window_frames(r, stream, W) if W is not None
                      else stream_of(r, stream)['per_frame'])
                frames += pf
                n_window.append(len(pf))
            if r.get('seg_vec_cos_to_full') is not None:
                cos_vals.append(r['seg_vec_cos_to_full'])
        if not per:
            continue
        rows.append({
            'prefix_fraction': f,
            'n_cases': len(per),
            'mean_frames_used': float(np.mean([r['n_frames_used'] for _, r in runs])),
            'mean_evaluated_frames': float(np.mean(n_window)) if n_window else None,
            'seg_vec_cos_to_full': float(np.mean(cos_vals)) if cos_vals else None,
            'case_macro': {
                k: agg([p[k] for p in per], bootstrap, seed)
                for k in ('iou_target', 'iou_distractor', 'margin', 'id_err')},
            'decisions': {k: sum(1 for p in per if p['identity_decision'] == k)
                          for k in ('target', 'distractor', 'empty')},
            'frame_micro': micro(frames),
        })
    return rows


def paired(recs, lo_frac, hi_frac, stream, W_of, seed, bootstrap):
    """Paired low-vs-high evidence comparison on the identical frame set."""
    lo_err, hi_err, lo_marg, hi_marg, frames_lo, frames_hi = [], [], [], [], [], []
    corr = {'wrong_fixed': 0, 'wrong_persist': 0, 'right_right': 0, 'right_broken': 0}
    flips, rows, wins = 0, [], []
    for cid, rec in recs.items():
        r_lo, r_hi = get_run(rec, PREFIX_MODE, lo_frac), get_run(rec, PREFIX_MODE, hi_frac)
        if not r_lo or not r_hi:
            continue
        W = W_of(cid)
        m_lo, m_hi = case_metrics(r_lo, stream, W), case_metrics(r_hi, stream, W)
        if not m_lo or not m_hi:
            continue
        pf_lo, pf_hi = window_frames(r_lo, stream, W), window_frames(r_hi, stream, W)
        if len(pf_lo) != len(pf_hi):
            raise SystemExit(f'{cid}: window lengths differ ({len(pf_lo)} vs {len(pf_hi)}) '
                             f'— the common-window rule is broken')
        lo_err.append(m_lo['id_err']); hi_err.append(m_hi['id_err'])
        lo_marg.append(m_lo['margin']); hi_marg.append(m_hi['margin'])
        frames_lo += pf_lo; frames_hi += pf_hi
        wins.append(W)
        wrong_lo, wrong_hi = m_lo['margin'] < 0, m_hi['margin'] < 0
        if wrong_lo and not wrong_hi:
            corr['wrong_fixed'] += 1
        elif wrong_lo and wrong_hi:
            corr['wrong_persist'] += 1
        elif not wrong_lo and not wrong_hi:
            corr['right_right'] += 1
        else:
            corr['right_broken'] += 1
        if m_lo['identity_decision'] != m_hi['identity_decision']:
            flips += 1
        rows.append({'case_id': cid, 'window_frames': W,
                     'id_err_lo': m_lo['id_err'], 'id_err_hi': m_hi['id_err'],
                     'margin_lo': m_lo['margin'], 'margin_hi': m_hi['margin'],
                     'decision_lo': m_lo['identity_decision'],
                     'decision_hi': m_hi['identity_decision'],
                     'delta_margin_hi_minus_lo': m_hi['margin'] - m_lo['margin']})
    return {
        'lo_prefix': lo_frac, 'hi_prefix': hi_frac,
        'n_cases': len(rows),
        'window_frames': {'min': int(min(wins)) if wins else None,
                          'median': float(np.median(wins)) if wins else None,
                          'max': int(max(wins)) if wins else None},
        'id_err_lo': agg(lo_err, bootstrap, seed), 'id_err_hi': agg(hi_err, bootstrap, seed),
        'delta_id_err_hi_minus_lo': paired_delta(lo_err, hi_err, seed, bootstrap),
        'margin_lo': agg(lo_marg, bootstrap, seed), 'margin_hi': agg(hi_marg, bootstrap, seed),
        'delta_margin_hi_minus_lo': paired_delta(lo_marg, hi_marg, seed, bootstrap),
        'frame_micro_lo': micro(frames_lo), 'frame_micro_hi': micro(frames_hi),
        'error_transitions': corr, 'decision_flip_cases': flips,
        'frames_per_case_identical': True,
        'per_case': rows,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs, provenance = load_records(args.audit)
    fr = sorted({r['prefix_fraction'] for rec in recs.values()
                 for r in rec['runs'] if r['mode'] == PREFIX_MODE})
    lo = args.paired_lo if args.paired_lo is not None else min(fr)
    hi = args.paired_hi if args.paired_hi is not None else max(fr)

    # ---- strict per-case common window ----
    def W_of(cid, fracs=None):
        rec = recs[cid]
        use = fr if fracs is None else fracs
        lens = [r['sam2_prompt_frames'] for r in rec['runs']
                if r['mode'] == PREFIX_MODE and r['prefix_fraction'] in use]
        return int(min([args.window] + lens)) if lens else args.window

    windows = {cid: W_of(cid) for cid in recs}
    fallback_cases = sorted(cid for cid, rec in recs.items()
                            if any(r['identity_stream_used'] != 'raw' for r in rec['runs']))

    out = {
        'schema_version': 2,
        'n_cases': len(recs),
        'prefixes': fr,
        'audit_files': provenance,
        'primary_stream': 'raw',
        'secondary_stream': 'gated',
        'identity_stream_fallback_cases': fallback_cases,
        'window_rule': 'W_i = min(over compared runs) of sam2_prompt_frames, capped at '
                       f'{args.window}; both sides use frames 0..W_i-1 of case i',
        'window_len': {'min': int(min(windows.values())) if windows else None,
                       'median': float(np.median(list(windows.values()))) if windows else None,
                       'max': int(max(windows.values())) if windows else None},
        'per_case_window': windows,
    }

    # ---- (a) confounded: each prefix over its own horizon ----
    out['full_horizon_curve_raw'] = curve(recs, fr, 'raw', None, args.bootstrap, args.seed)
    out['full_horizon_curve_raw_note'] = (
        'CONFOUNDED across prefixes: each prefix averages over its own (different) '
        'time interval, so a shorter prefix is not a fair comparison. Kept only for '
        'transparency / continuity with the v1 report; never quote as evidence.')

    # ---- (b) primary: every prefix on the identical per-case window ----
    out['common_window_curve_raw'] = curve(recs, fr, 'raw', W_of, args.bootstrap, args.seed)
    out['common_window_curve_gated'] = curve(recs, fr, 'gated', W_of, args.bootstrap, args.seed)

    out['paired_common_window_raw'] = paired(recs, lo, hi, 'raw', W_of,
                                             args.seed, args.bootstrap)
    out['paired_common_window_gated'] = paired(recs, lo, hi, 'gated', W_of,
                                              args.seed, args.bootstrap)

    # ---- (c) default path (VLM sees only the first 5 frames) on the same window ----
    d_runs = [(cid, get_run(rec, DEFAULT_MODE)) for cid, rec in recs.items()]
    d_runs = [(cid, r) for cid, r in d_runs if r]
    # Controlled A/B with NO horizon confound at all: `full_video_default_vlm_first5`
    # and `prefix 1.0` have the same SAM2 prompt frames (min(5,T)) and the same
    # propagation horizon (the whole video); the ONLY difference is whether the
    # VLM saw every frame or only the first 5. Window rule: min over both runs of
    # min(sam2_prompt_frames, n_frames_used), capped at --window.
    def W_default(cid):
        rec = recs[cid]
        lens = []
        for r in rec['runs']:
            if r['mode'] == DEFAULT_MODE or (r['mode'] == PREFIX_MODE
                                             and abs(r['prefix_fraction'] - max(fr)) < 1e-9):
                lens += [r['sam2_prompt_frames'], r['n_frames_used']]
        return int(min([args.window] + lens)) if lens else args.window

    dv_err, fv_err, dv_marg, fv_marg, dv_rows = [], [], [], [], []
    dv_frames, fv_frames = [], []
    for cid, rec in recs.items():
        r_d, r_f = get_run(rec, DEFAULT_MODE), get_run(rec, PREFIX_MODE, max(fr))
        if not r_d or not r_f:
            continue
        W = W_default(cid)
        m_d, m_f = case_metrics(r_d, 'raw', W), case_metrics(r_f, 'raw', W)
        if not m_d or not m_f:
            continue
        pf_d, pf_f = window_frames(r_d, 'raw', W), window_frames(r_f, 'raw', W)
        if len(pf_d) != len(pf_f):
            raise SystemExit(f'{cid}: default-vs-full window mismatch')
        dv_err.append(m_d['id_err']); fv_err.append(m_f['id_err'])
        dv_marg.append(m_d['margin']); fv_marg.append(m_f['margin'])
        dv_frames += pf_d; fv_frames += pf_f
        dv_rows.append({'case_id': cid, 'window_frames': W,
                        'id_err_vlm_first5': m_d['id_err'],
                        'id_err_vlm_all': m_f['id_err'],
                        'margin_vlm_first5': m_d['margin'],
                        'margin_vlm_all': m_f['margin'],
                        'decision_vlm_first5': m_d['identity_decision'],
                        'decision_vlm_all': m_f['identity_decision']})
    if dv_rows:
        out['controlled_vlm_first5_vs_vlm_all'] = {
            'design': 'identical SAM2 prompt frames (min(5,T)) and identical propagation '
                      'horizon (the whole video); the only difference is whether the VLM '
                      'saw every frame (prefix 1.0) or only the first 5 (default path)',
            'n_cases': len(dv_rows),
            'window_frames': {'min': int(min(r['window_frames'] for r in dv_rows)),
                              'median': float(np.median([r['window_frames'] for r in dv_rows])),
                              'max': int(max(r['window_frames'] for r in dv_rows))},
            'id_err_vlm_first5': agg(dv_err, args.bootstrap, args.seed),
            'id_err_vlm_all': agg(fv_err, args.bootstrap, args.seed),
            'delta_id_err_vlm_all_minus_first5': paired_delta(dv_err, fv_err,
                                                              args.seed, args.bootstrap),
            'margin_vlm_first5': agg(dv_marg, args.bootstrap, args.seed),
            'margin_vlm_all': agg(fv_marg, args.bootstrap, args.seed),
            'delta_margin_vlm_all_minus_first5': paired_delta(dv_marg, fv_marg,
                                                              args.seed, args.bootstrap),
            'frame_micro_vlm_first5': micro(dv_frames),
            'frame_micro_vlm_all': micro(fv_frames),
            'decision_flips': sum(1 for r in dv_rows
                                  if r['decision_vlm_first5'] != r['decision_vlm_all']),
            'per_case': dv_rows,
        }

    dm = [case_metrics(r, 'raw', W_of(cid, [lo, hi])) for cid, r in d_runs]
    dm = [m for m in dm if m]
    if dm:
        out['default_path_common_window_raw'] = {
            'n_cases': len(dm),
            'case_macro': {k: agg([m[k] for m in dm], args.bootstrap, args.seed)
                           for k in ('iou_target', 'iou_distractor', 'margin', 'id_err')},
            'decisions': {k: sum(1 for m in dm if m['identity_decision'] == k)
                          for k in ('target', 'distractor', 'empty')},
            'frame_micro': micro([f for cid, r in d_runs
                                  for f in window_frames(r, 'raw', W_of(cid, [lo, hi]))]),
            'note': 'vlm_all_frames=False on the full video, evaluated on the same '
                    'per-case window as the prefix runs',
        }

    # ---- (d) cosine calibration ----
    # ---- (e) how much does the verifier gate change the identity metrics? ----
    # Quantifies exactly why identity must be measured on raw masks: this is the
    # difference between "the segmenter picked the wrong instance" and "the
    # segmenter picked an instance and the verifier turned the frame off".
    rg_rows, rg_dec, rg_err = [], 0, []
    for cid, rec in recs.items():
        W = W_of(cid)
        for f in fr:
            run = get_run(rec, PREFIX_MODE, f)
            if not run:
                continue
            m_r, m_g = case_metrics(run, 'raw', W), case_metrics(run, 'gated', W)
            if not m_r or not m_g:
                continue
            if m_r['identity_decision'] != m_g['identity_decision']:
                rg_dec += 1
            rg_err.append(m_g['id_err'] - m_r['id_err'])
            rg_rows.append({'case_id': cid, 'prefix_fraction': f, 'window_frames': W,
                            'decision_raw': m_r['identity_decision'],
                            'decision_gated': m_g['identity_decision'],
                            'id_err_raw': m_r['id_err'], 'id_err_gated': m_g['id_err'],
                            'empty_rate_raw': m_r['empty_rate'],
                            'empty_rate_gated': m_g['empty_rate']})
    # case-level (cluster-respecting) comparison: mean over prefixes per case
    by_case = {}
    for r in rg_rows:
        by_case.setdefault(r['case_id'], {'raw': [], 'gated': []})
        by_case[r['case_id']]['raw'].append(r['id_err_raw'])
        by_case[r['case_id']]['gated'].append(r['id_err_gated'])
    case_raw = [float(np.mean(v['raw'])) for v in by_case.values()]
    case_gated = [float(np.mean(v['gated'])) for v in by_case.values()]
    out['raw_vs_gated_common_window'] = {
        'n_runs_compared': len(rg_rows),
        'decision_disagreements': rg_dec,
        'id_err_raw_case_macro': agg(case_raw, args.bootstrap, args.seed),
        'id_err_gated_case_macro': agg(case_gated, args.bootstrap, args.seed),
        'id_err_gated_minus_raw_case_paired': paired_delta(case_raw, case_gated,
                                                           args.seed, args.bootstrap),
        'empty_rate_raw_mean': float(np.mean([r['empty_rate_raw'] for r in rg_rows])) if rg_rows else None,
        'empty_rate_gated_mean': float(np.mean([r['empty_rate_gated'] for r in rg_rows])) if rg_rows else None,
        'note': 'an empty (gated-off) frame scores id_err=0 under the IoU comparison, so '
                'the gated stream understates identity errors exactly when the verifier '
                'abstains — which is why identity is reported on raw masks',
        'per_run': rg_rows,
    }

    c2full = [r['seg_vec_cos_to_full'] for rec in recs.values() for r in rec['runs']
              if r['mode'] == PREFIX_MODE and r['seg_vec_cos_to_full'] is not None]
    oq = [rec['calib']['cos_to_full_prefix_segvec'] for rec in recs.values()
          if rec.get('calib') and rec['calib'].get('cos_to_full_prefix_segvec') is not None]
    out['cos_calibration'] = {
        'prefix_to_full_same_query': {
            'n': len(c2full), 'mean': float(np.mean(c2full)) if c2full else None,
            'min': float(np.min(c2full)) if c2full else None,
            'max': float(np.max(c2full)) if c2full else None},
        'different_query_same_video_full_prefix': {
            'n': len(oq), 'mean': float(np.mean(oq)) if oq else None,
            'median': float(np.median(oq)) if oq else None,
            'min': float(np.min(oq)) if oq else None,
            'max': float(np.max(oq)) if oq else None},
        'interpretation_rule': 'a prefix-to-full cos is only informative relative to '
                               'the different-query cos on the same videos: if the two '
                               'scales are comparable, the cosine does not measure '
                               'referent revision',
        'per_case': [{'case_id': rec['case_id'],
                      'other_query': rec['calib']['other_query'],
                      'cos_other_query': rec['calib']['cos_to_full_prefix_segvec']}
                     for rec in recs.values() if rec.get('calib')],
    }
    out['caveats'] = [
        'identity is computed on RAW masks (pre-gate); the gated stream is reported '
        'separately and must not be quoted as identity fidelity',
        'all cross-prefix numbers use the strict per-case common window; the '
        'full-horizon curve is retained but labelled confounded',
    ]

    with open(os.path.join(args.out_dir, 'prefix_identity_eval_v2.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False)

    if HAVE_MPL:
        xs = [r['mean_evaluated_frames'] or r['mean_frames_used']
              for r in out['common_window_curve_raw']]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        y = [r['case_macro']['id_err']['mean'] for r in out['common_window_curve_raw']]
        lo_e = [r['case_macro']['id_err']['mean'] - r['case_macro']['id_err']['lo']
                for r in out['common_window_curve_raw']]
        hi_e = [r['case_macro']['id_err']['hi'] - r['case_macro']['id_err']['mean']
                for r in out['common_window_curve_raw']]
        ax[0].errorbar(xs, y, yerr=[lo_e, hi_e], marker='o',
                       label='IDErr (raw, common window)')
        ax[0].plot(xs, [r['case_macro']['iou_target']['mean']
                        for r in out['common_window_curve_raw']], marker='s',
                   label='IoU target')
        ax[0].plot(xs, [r['case_macro']['iou_distractor']['mean']
                        for r in out['common_window_curve_raw']], marker='^',
                   label='IoU distractor')
        ax[0].set_xlabel('frames of evidence given to the VLM')
        ax[0].set_title('identity vs evidence volume\n(raw masks, identical per-case window)')
        ax[0].legend(); ax[0].grid(alpha=.3)
        vals = [{'x': r['mean_frames_used'], 'y': r['seg_vec_cos_to_full']}
                for r in out['full_horizon_curve_raw'] if r.get('seg_vec_cos_to_full')]
        if vals:
            ax[1].plot([v['x'] for v in vals], [v['y'] for v in vals], marker='o',
                       label='prefix vs full (same query)')
        cal = out['cos_calibration']['different_query_same_video_full_prefix']['mean']
        if cal is not None:
            ax[1].axhline(cal, ls='--', color='crimson',
                          label=f'different query (mean {cal:.3f})')
        ax[1].set_ylim(0.5, 1.02)
        ax[1].set_xlabel('frames of evidence')
        ax[1].set_title('cos([SEG]) vs evidence volume, with query-change scale')
        ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'prefix_identity_curve_v2.png'), dpi=160)
        print(f'[prefix-eval] plot -> {args.out_dir}/prefix_identity_curve_v2.png', flush=True)

    write_run_metadata(
        os.path.join(args.out_dir, 'prefix_identity_eval_v2.meta.json'),
        command=' '.join(sys.argv), script=os.path.abspath(__file__),
        git_sha=git_sha(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))),
        audit_files=[p['path'] for p in provenance],
        model=DEFAULT_MODEL, dataset='Ref-YT-VOS valid',
        ann_root=DEFAULT_ANN_ROOT, jpeg_root=DEFAULT_JPEG_ROOT,
        manifest=DEFAULT_MANIFEST,
        seed=args.seed, bootstrap_resamples=args.bootstrap,
        primary_stream='raw', secondary_stream='gated',
        window_cap=args.window, paired_lo=lo, paired_hi=hi,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'none'), world_size=1,
        new_inference=False,
        note='consumes the v2 audit JSON (raw_masks + gated); no model is loaded',
        n_cases=len(recs), identity_stream_fallback_cases=fallback_cases)

    pw = out['paired_common_window_raw']
    print(f'[prefix-eval v2] cases={len(recs)} window(min/med/max)='
          f'{out["window_len"]["min"]}/{out["window_len"]["median"]}/{out["window_len"]["max"]}',
          flush=True)
    print(f'[prefix-eval v2] RAW paired {lo}->{hi} on identical window: '
          f'IDErr {pw["id_err_lo"]["mean"]:.3f} -> {pw["id_err_hi"]["mean"]:.3f} '
          f'(delta {pw["delta_id_err_hi_minus_lo"]["mean_delta"]:+.3f} '
          f'[{pw["delta_id_err_hi_minus_lo"]["lo"]:+.3f},'
          f'{pw["delta_id_err_hi_minus_lo"]["hi"]:+.3f}]) '
          f'corrections {pw["error_transitions"]["wrong_fixed"]}/'
          f'{pw["error_transitions"]["wrong_fixed"] + pw["error_transitions"]["wrong_persist"]}',
          flush=True)
    print(f'[prefix-eval v2] cos calibration: same-query prefix->full mean='
          f'{out["cos_calibration"]["prefix_to_full_same_query"]["mean"]:.3f} | '
          f'different-query mean='
          f'{out["cos_calibration"]["different_query_same_video_full_prefix"]["mean"]:.3f}',
          flush=True)
    print(f'[prefix-eval v2] wrote {args.out_dir}/prefix_identity_eval_v2.json', flush=True)


if __name__ == '__main__':
    main()

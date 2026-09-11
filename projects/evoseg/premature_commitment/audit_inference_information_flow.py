"""M5.0 — inference information-flow audit (does NOT change model behaviour).

Runs the SAME video/query through the real EvoSeg/Sa2VA inference path under:

  * prefix fractions 20/40/60/80/100% of the annotated frames, with
    `vlm_all_frames=True` (the setting used by the official Ref-YT-VOS runner);
  * the default VLM path (`vlm_all_frames=False`, i.e. only the first 5 frames
    reach the VLM) on the full video, for the full-video-vs-prefix comparison.

Two mask streams are recorded for every run, and they are NOT the same thing:

  raw    — the SAM2 propagation output before the temporal verifier gate
           (`model._raw_masks`), i.e. segmentation + tracking only;
  gated  — the final system output (`prediction_masks`), i.e. raw * e_t.

Identity analysis uses **raw**, because the gated stream mixes referent identity
with the verifier's abstention: a gated-off frame is empty, and an empty frame
counts as "no identity error" under the IoU comparison. The gated numbers are
kept as a secondary, clearly labelled result.

For every run it also records: decoded text, whether `[SEG]` was emitted, the
projected `[SEG]` / segmentation-conditioning vector, cosine similarity of that
vector against the 100%-prefix vector, per-frame target IoU / max-distractor
IoU / identity margin / IDErr / empty flag, the resulting identity decision, and
the SAM2 prompt/propagation frame counts the architecture implies (prompt on the
first min(5, N) frames, propagate over all).

Optionally (default on) one extra run per case uses a *different* expression of
the same video, to calibrate what a given cosine similarity between two `[SEG]`
vectors means.

Output schema version 2 (see `schema_version` in the JSON).

Nothing here is tuned on GT: GT masks are used ONLY to compute metrics.

Usage
-----
python audit_inference_information_flow.py \
  --model /path/to/EvoSeg-Qwen3-VL-4B-Faithful \
  --manifest /path/to/faithfulness_valid.json \
  --out-dir docs/premature_commitment/results \
  --n-cases 20 --prefixes 0.2 0.4 0.6 0.8 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    DEFAULT_ANN_ROOT, DEFAULT_ARTIFACT_ROOT, DEFAULT_JPEG_ROOT, DEFAULT_MANIFEST,
    DEFAULT_META, DEFAULT_MODEL, DEFAULT_RYVOS_ROOT, N_DEFAULT_PROMPT_FRAMES,
    case_instance_masks, git_sha, load_frames, load_manifest, load_meta,
    per_frame_identity, write_run_metadata,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='HF model dir (must contain the EvoSeg/Sa2VA remote code)')
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--jpeg-root', default=DEFAULT_JPEG_ROOT)
    ap.add_argument('--ann-root', default=DEFAULT_ANN_ROOT)
    ap.add_argument('--meta', default=DEFAULT_META)
    ap.add_argument('--artifact-root', default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument('--out-dir', default='docs/premature_commitment/results')
    ap.add_argument('--category', default='identity_swap',
                    help='manifest category to audit (same-class multi-instance by construction)')
    ap.add_argument('--n-cases', type=int, default=20)
    ap.add_argument('--prefixes', type=float, nargs='+',
                    default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument('--min-prefix-frames', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true',
                    help='run one case only, for a quick sanity check')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shard-index', type=int, default=0,
                    help='run only this shard of the selected cases (for a multi-GPU '
                         'reproduction check); recorded in the metadata')
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--tag', default=None,
                    help='override the output filename tag (default: n<#cases>)')
    ap.add_argument('--no-calib-other-query', action='store_true',
                    help='skip the extra run with a different expression of the same '
                         'video used to calibrate [SEG] cosine similarity')
    return ap.parse_args()


def mask_stream(obj, n_frames, label):
    """[T,H,W] tensor / ndarray -> (list of bool masks, available flag, reason).

    A stream that is missing, or whose frame count does not match the video, is
    reported explicitly instead of being silently reused (a stale `_raw_masks`
    from a previous call would look plausible but belong to another video).
    """
    if obj is None:
        return [None] * n_frames, False, f'{label}: not returned by predict_forward'
    arr = obj.detach().cpu().numpy() if hasattr(obj, 'detach') else np.asarray(obj)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return [None] * n_frames, False, f'{label}: unexpected shape {arr.shape}'
    if arr.shape[0] != n_frames:
        return [None] * n_frames, False, (
            f'{label}: {arr.shape[0]} frames for a {n_frames}-frame input '
            f'(stale buffer?) — treated as unavailable')
    return [arr[t] > 0 for t in range(n_frames)], True, None


def select_cases(cases, category, n, seed):
    pool = [c for c in cases if c.get('category') == category]
    if not pool:
        raise RuntimeError(f'no cases of category={category} in manifest')
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))[:n]
    return [pool[i] for i in sorted(idx)]


def run_once(model, tok, proc, frames, query, vlm_all_frames):
    """One forward pass. Returns a record plus the [SEG] vector and masks."""
    text = f'<image>\n Please segment {query} in this video.'
    info = {'vlm_all_frames_requested': bool(vlm_all_frames),
            'vlm_all_frames_fallback': False}
    try:
        with torch.no_grad():
            out = model.predict_forward(video=frames, text=text, tokenizer=tok,
                                        processor=proc, vlm_all_frames=vlm_all_frames)
    except TypeError as exc:
        # explicit, recorded fallback (never silent): the official runner has the
        # same one; it means this checkpoint's remote code lacks the flag.
        info['vlm_all_frames_fallback'] = True
        info['vlm_all_frames_error'] = str(exc)
        print(f'  [fallback] vlm_all_frames unsupported ({exc}); using default path', flush=True)
        with torch.no_grad():
            out = model.predict_forward(video=frames, text=text, tokenizer=tok,
                                        processor=proc)

    pred_text = out.get('prediction', '')
    has_seg = '[SEG]' in pred_text or '[seg]' in pred_text.lower()
    lang = out.get('lang')
    seg_vec = None
    if lang is not None:
        seg_vec = np.asarray(lang).reshape(-1).astype(np.float32)
    iou_state = getattr(model, 'existence_head', None)  # placeholder for optional heads
    info['has_seg'] = bool(has_seg)
    info['text'] = pred_text[:300]
    info['seg_vec_dim'] = int(seg_vec.shape[0]) if seg_vec is not None else 0
    info['seg_vec'] = seg_vec.tolist() if seg_vec is not None else None
    info['iou_state_head_present'] = iou_state is not None

    pm = out.get('prediction_masks')
    n_t = len(frames)
    gated, gated_ok, gated_reason = mask_stream(
        pm[0] if pm else None, n_t, 'prediction_masks (gated)')
    raw, raw_ok, raw_reason = mask_stream(out.get('raw_masks'), n_t, 'raw_masks')
    if not raw_ok:
        # Explicit, recorded fallback: without the pre-gate stream the identity
        # metrics would silently become "gated identity", which is a different
        # quantity. Say so instead.
        print(f'  [warn] raw mask stream unavailable ({raw_reason}); '
              f'identity metrics fall back to the GATED stream for this run', flush=True)
    info['n_masks'] = len(gated)
    info['n_seg_tokens'] = int(len(pm)) if pm else 0
    info['raw_available'] = bool(raw_ok)
    info['raw_unavailable_reason'] = raw_reason
    info['gated_available'] = bool(gated_ok)
    info['gated_unavailable_reason'] = gated_reason
    info['e_logit'] = None
    el = out.get('e_logit')
    if el is not None:
        el = el.detach().cpu().numpy() if hasattr(el, 'detach') else np.asarray(el)
        info['e_logit'] = np.asarray(el).reshape(-1).astype(float).tolist()
    return info, raw, gated


def audit_case(case, model, tok, proc, args, meta):
    T = len(case['frames'])
    frames_all = load_frames(case, args.jpeg_root)
    tgt_obj, inst_masks = case_instance_masks(case, args.ann_root, meta)
    if tgt_obj is None:
        raise RuntimeError(f'case {case["video_id"]}/{case.get("exp_id")} has no target obj_id')

    rec = {
        'case_id': f'{case["video_id"]}:{case.get("exp_id")}',
        'video_id': case['video_id'],
        'exp_id': case.get('exp_id'),
        'target_obj_id': tgt_obj,
        'query': case['query'],
        'n_frames_total': T,
        'candidate_obj_ids': sorted(inst_masks.keys()),
        'n_candidate_instances': len(inst_masks),
        'runs': [],
    }

    def stream_metrics(masks, n_used):
        """Per-frame identity metrics for one mask stream (raw or gated)."""
        per_frame = []
        first_nonempty = None
        for t in range(n_used):
            tgt = inst_masks[tgt_obj][t]
            dists = [inst_masks[o][t] for o in inst_masks if o != tgt_obj]
            st = per_frame_identity(masks[t] if t < len(masks) else None, tgt, dists)
            st['t'] = t
            st['gt_target_present'] = bool(tgt.any()) if tgt is not None else False
            if first_nonempty is None and not st['empty']:
                first_nonempty = t
            per_frame.append(st)
        nonempty = [r for r in per_frame if not r['empty']]
        return {
            'first_nonempty_frame': first_nonempty,
            'n_nonempty_frames': len(nonempty),
            'iou_target_mean': float(np.mean([r['iou_target'] for r in per_frame])) if per_frame else None,
            'iou_distractor_mean': float(np.mean([r['iou_distractor_max'] for r in per_frame])) if per_frame else None,
            'identity_margin_mean': float(np.mean([r['identity_margin'] for r in per_frame])) if per_frame else None,
            'id_err_rate': float(np.mean([r['id_err'] for r in per_frame])) if per_frame else None,
            'empty_rate': float(np.mean([r['empty'] for r in per_frame])) if per_frame else None,
            'identity_decision': ('empty' if not nonempty else
                                  'target' if np.mean([r['identity_margin'] for r in nonempty]) >= 0 else
                                  'distractor'),
            'per_frame': per_frame,
        }

    def summarise(info, raw_masks, gated_masks, n_used, prefix_frac, mode, other_query=None):
        d = {
            'mode': mode,
            'prefix_fraction': prefix_frac,
            'n_frames_used': n_used,
            'has_seg': info['has_seg'],
            'text': info['text'],
            'vlm_all_frames_requested': info['vlm_all_frames_requested'],
            'vlm_all_frames_fallback': info['vlm_all_frames_fallback'],
            'seg_vec': info['seg_vec'],
            'seg_vec_dim': info['seg_vec_dim'],
            'sam2_prompt_frames': int(min(N_DEFAULT_PROMPT_FRAMES, n_used)),
            'sam2_propagation_frames': int(n_used),
            'other_query': other_query,
            'raw_available': info['raw_available'],
            'raw_unavailable_reason': info['raw_unavailable_reason'],
            'gated_available': info['gated_available'],
            'e_logit': info['e_logit'],
            # PRIMARY stream: segmentation + tracking, before the verifier gate.
            'raw': stream_metrics(raw_masks, n_used),
            # SECONDARY stream: final system output (raw * e_t), mixes identity
            # with the verifier's abstention — never used for identity claims.
            'gated': stream_metrics(gated_masks, n_used) if info['gated_available'] else None,
        }
        return d

    # ---- prefix runs (all frames to the VLM within the prefix) ----
    def add_run(info, raw, gated, n_used, frac, mode, other_query=None):
        """Append one run; falls back to the gated stream only if raw is missing."""
        if info['raw_available']:
            use_raw, stream_used = raw, 'raw'
        else:
            use_raw, stream_used = gated, 'gated_fallback'
        d = summarise(info, use_raw, gated, n_used, frac, mode, other_query=other_query)
        d['identity_stream_used'] = stream_used
        rec['runs'].append(d)
        return d

    for frac in sorted(args.prefixes):
        n_used = max(args.min_prefix_frames, int(round(frac * T)))
        n_used = min(n_used, T)
        info, raw, gated = run_once(model, tok, proc, frames_all[:n_used], case['query'],
                                    vlm_all_frames=True)
        d = add_run(info, raw, gated, n_used, float(frac), 'prefix_vlm_all_frames')
        print(f'  [{rec["case_id"]}] prefix={frac:.2f} n={n_used} seg={info["has_seg"]} '
              f'id={d["raw"]["identity_decision"]} '
              f'iou_t={d["raw"]["iou_target_mean"]:.4f} '
              f'iou_d={d["raw"]["iou_distractor_mean"]:.4f} '
              f'stream={d["identity_stream_used"]}', flush=True)

    # ---- default VLM path on the full video (only first 5 frames reach the VLM) ----
    info, raw, gated = run_once(model, tok, proc, frames_all, case['query'],
                                vlm_all_frames=False)
    d = add_run(info, raw, gated, T, 1.0, 'full_video_default_vlm_first5')
    print(f'  [{rec["case_id"]}] default(first-5 VLM) '
          f'id={d["raw"]["identity_decision"]} iou_t={d["raw"]["iou_target_mean"]:.4f} '
          f'stream={d["identity_stream_used"]}', flush=True)

    # ---- calibration run: a DIFFERENT expression of the same video ----
    # Gives the scale for "how different are two [SEG] vectors when the query is
    # really different", without which a cosine of 0.96 is uninterpretable.
    rec['calib'] = None
    if not args.no_calib_other_query:
        exps = meta[case['video_id']]['expressions']
        others = [(str(eid), e['exp']) for eid, e in exps.items()
                  if str(eid) != str(case.get('exp_id'))]
        if others:
            oid, oquery = others[0]
            info_c, raw_c, gated_c = run_once(model, tok, proc, frames_all, oquery,
                                              vlm_all_frames=False)
            v_c = info_c.get('seg_vec')
            v_full = None
            for r in rec['runs']:
                if r['mode'] == 'full_video_default_vlm_first5':
                    v_full = r['seg_vec']
            cos_oq = None
            if v_c is not None and v_full is not None:
                a = np.asarray(v_c, dtype=np.float32)
                b = np.asarray(v_full, dtype=np.float32)
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                cos_oq = float(np.dot(a, b) / den) if den > 0 else None
            rec['calib'] = {
                'other_exp_id': oid, 'other_query': oquery,
                'same_frames': True, 'vlm_all_frames': False,
                'cos_to_full_prefix_segvec': cos_oq,
                'seg_vec_dim': info_c.get('seg_vec_dim'),
            }
            print(f'  [{rec["case_id"]}] calib cos(other query)={cos_oq}', flush=True)
        else:
            print(f'  [{rec["case_id"]}] calib skipped: no other expression in video',
                  flush=True)

    # ---- cosine similarity of the [SEG] vector vs the 100% prefix ----
    full = [r for r in rec['runs'] if r['mode'] == 'prefix_vlm_all_frames'
            and abs(r['prefix_fraction'] - 1.0) < 1e-9]
    ref = np.asarray(full[0]['seg_vec'], dtype=np.float32) if full and full[0]['seg_vec'] else None
    for r in rec['runs']:
        v = r.get('seg_vec')
        if ref is None or v is None:
            r['seg_vec_cos_to_full'] = None
            continue
        v = np.asarray(v, dtype=np.float32)
        denom = float(np.linalg.norm(v) * np.linalg.norm(ref))
        cos = float(np.dot(v, ref) / denom) if denom > 0 else None
        r['seg_vec_cos_to_full'] = cos
        r['seg_vec_l2_to_full'] = float(np.linalg.norm(v - ref))
    return rec


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = args.model
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f'model dir not found: {model_path}')
    cases_all = load_manifest(args.manifest)
    meta = load_meta(args.meta)
    cases = select_cases(cases_all, args.category, 1 if args.dry_run else args.n_cases, args.seed)
    n_selected = len(cases)
    if args.shard_count > 1:
        if not (0 <= args.shard_index < args.shard_count):
            raise ValueError(f'--shard-index must be in [0, {args.shard_count})')
        cases = [c for i, c in enumerate(cases) if i % args.shard_count == args.shard_index]
        print(f'[audit] shard {args.shard_index}/{args.shard_count}: '
              f'{len(cases)} of {n_selected} selected cases', flush=True)

    print(f'[audit] model={model_path}', flush=True)
    print(f'[audit] {len(cases)} cases of category={args.category}', flush=True)
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, use_flash_attn=True,
                                      trust_remote_code=True).eval().to(args.device)
    head_loaded = False
    if hasattr(model, 'load_temporal_head'):
        try:
            model.load_temporal_head()
            head_loaded = True
        except Exception as exc:  # recorded, not swallowed
            print(f'[audit] load_temporal_head failed: {exc}', flush=True)
    print(f'[audit] temporal head loaded: {head_loaded}', flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    records = []
    for i, case in enumerate(cases):
        print(f'[audit] case {i+1}/{len(cases)} {case["video_id"]}', flush=True)
        records.append(audit_case(case, model, tok, proc, args, meta))

    out = {
        'schema_version': 2,
        'category': args.category,
        'model_path': model_path,
        'manifest': args.manifest,
        'prefix_fractions': args.prefixes,
        'temporal_head_loaded': head_loaded,
        'identity_stream': 'raw_masks (pre-gate SAM2 propagation)',
        'secondary_stream': 'gated prediction_masks (raw * e_t), recorded but never '
                            'used for identity claims',
        'calib_other_query': not args.no_calib_other_query,
        'records': records,
    }
    tag = args.tag or ('dryrun' if args.dry_run else f'n{len(cases)}')
    out_path = os.path.join(args.out_dir, f'P0_prefix_audit_{args.category}_{tag}.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f'[audit] wrote {out_path}', flush=True)

    write_run_metadata(os.path.join(args.out_dir, f'P0_prefix_audit_{args.category}_{tag}.meta.json'),
                       git_sha=git_sha(os.path.dirname(os.path.dirname(os.path.dirname(
                           os.path.dirname(os.path.abspath(__file__)))))),
                       command=' '.join(sys.argv),
                       model_path=model_path, dataset_root=args.artifact_root,
                       manifest_path=args.manifest, seed=args.seed,
                       world_size=1, n_cases=len(cases), category=args.category,
                       prefix_fractions=args.prefixes,
                       # flags are per run mode, not global: the prefix runs set
                       # vlm_all_frames=True and the default-path run sets False
                       run_modes=[
                           {'mode': 'prefix_vlm_all_frames', 'prefix_fraction': float(f),
                            'vlm_all_frames': True} for f in sorted(args.prefixes)
                       ] + [{'mode': 'full_video_default_vlm_first5', 'prefix_fraction': 1.0,
                             'vlm_all_frames': False}],
                       vlm_all_frames_prefix_runs=True,
                       vlm_all_frames_default_path=False,
                       device=args.device,
                       gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                       shard_index=args.shard_index, shard_count=args.shard_count,
                       n_cases_selected_before_sharding=n_selected,
                       case_ids=[c['video_id'] + ':' + str(c['exp_id']) for c in cases],
                       temporal_head_loaded=head_loaded,
                       schema_version=2,
                       identity_stream='raw_masks',
                       secondary_stream='gated prediction_masks',
                       calib_other_query=not args.no_calib_other_query,
                       n_runs_per_case=len(args.prefixes) + 2,
                       output=out_path)


if __name__ == '__main__':
    main()

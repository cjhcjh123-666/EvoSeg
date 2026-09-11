"""M5.0 — inference information-flow audit (does NOT change model behaviour).

Runs the SAME video/query through the real EvoSeg/Sa2VA inference path under:

  * prefix fractions 20/40/60/80/100% of the annotated frames, with
    `vlm_all_frames=True` (the setting used by the official Ref-YT-VOS runner);
  * the default VLM path (`vlm_all_frames=False`, i.e. only the first 5 frames
    reach the VLM) on the full video, for the full-video-vs-prefix comparison.

For every run it records: decoded text, whether `[SEG]` was emitted, the
projected `[SEG]` / segmentation-conditioning vector, cosine similarity of that
vector against the 100%-prefix vector, first non-empty mask frame, per-frame
target IoU / max-distractor IoU / identity margin / IDErr / empty flag, the
resulting identity decision, and the SAM2 prompt/propagation frame counts the
architecture implies (prompt on the first min(5, N) frames, propagate over all).

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
    return ap.parse_args()


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
    masks = []
    if pm:
        pred = pm[0]
        masks = [np.asarray(pred[t]) > 0 if t < len(pred) else None
                 for t in range(len(frames))]
    else:
        masks = [None] * len(frames)
    info['n_masks'] = len(masks)
    info['n_seg_tokens'] = int(len(pm)) if pm else 0
    return info, masks


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

    def summarise(info, masks, n_used, prefix_frac, mode):
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
            'sam2_prompt_frames': int(min(N_DEFAULT_PROMPT_FRAMES, n_used)),
            'sam2_propagation_frames': int(n_used),
            'per_frame': per_frame,
        }
        return d

    # ---- prefix runs (all frames to the VLM within the prefix) ----
    for frac in sorted(args.prefixes):
        n_used = max(args.min_prefix_frames, int(round(frac * T)))
        n_used = min(n_used, T)
        info, masks = run_once(model, tok, proc, frames_all[:n_used], case['query'],
                               vlm_all_frames=True)
        rec['runs'].append(summarise(info, masks, n_used, float(frac), 'prefix_vlm_all_frames'))
        print(f'  [{rec["case_id"]}] prefix={frac:.2f} n={n_used} seg={info["has_seg"]} '
              f'id={rec["runs"][-1]["identity_decision"]} '
              f'iou_t={rec["runs"][-1]["iou_target_mean"]} '
              f'iou_d={rec["runs"][-1]["iou_distractor_mean"]}', flush=True)

    # ---- default VLM path on the full video (only first 5 frames reach the VLM) ----
    info, masks = run_once(model, tok, proc, frames_all, case['query'], vlm_all_frames=False)
    rec['runs'].append(summarise(info, masks, T, 1.0, 'full_video_default_vlm_first5'))
    print(f'  [{rec["case_id"]}] default(first-5 VLM) id={rec["runs"][-1]["identity_decision"]} '
          f'iou_t={rec["runs"][-1]["iou_target_mean"]}', flush=True)

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
        'category': args.category,
        'model_path': model_path,
        'manifest': args.manifest,
        'prefix_fractions': args.prefixes,
        'temporal_head_loaded': head_loaded,
        'records': records,
    }
    tag = 'dryrun' if args.dry_run else f'n{len(cases)}'
    out_path = os.path.join(args.out_dir, f'P0_prefix_audit_{args.category}_{tag}.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f'[audit] wrote {out_path}', flush=True)

    write_run_metadata(os.path.join(args.out_dir, f'P0_prefix_audit_{args.category}_{tag}.meta.json'),
                       git_sha='/9950backfile/chenjiahui/EvoSeg',
                       command=' '.join(sys.argv),
                       model_path=model_path, dataset_root=args.artifact_root,
                       manifest_path=args.manifest, seed=args.seed,
                       world_size=1, n_cases=len(cases), category=args.category,
                       prefix_fractions=args.prefixes, vlm_all_frames=True,
                       temporal_head_loaded=head_loaded,
                       output=out_path)


if __name__ == '__main__':
    main()

"""Generate side-by-side video examples for the EvoSeg temporal faithfulness fix.

For a few curated faithfulness_valid cases, run the SAME model twice:
  * "before" : TEG-4B per-frame MLP e_t gate (temporal_existence_head=None)
  * "after"  : TEG-4B + trained GRU temporal existence head
and render a video with:
  * query text (the "question language" used by the benchmark)
  * per-frame GT presence badge (PRESENT / ABSENT)
  * green mask overlay on segmented pixels
  * left=before, right=after, with per-frame predicted-e_t strip

Usage:
  CUDA_VISIBLE_DEVICES=0 python make_temporal_example_videos.py \
      --out /9950backfile/chenjiahui/evo_artifacts/data/examples
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/valid/JPEGImages')
MANIFEST = ('/9950backfile/chenjiahui/evo_artifacts/datasets/'
            'ref_youtube_vos/faithfulness_valid.json')

# curated: (index_in_manifest, title) — pick representative hard cases
CURATED = [
    (64, 'temporal_clean_stop'),
    (184, 'temporal_reappear'),
    (18, 'identity_swap'),
    (0, 'global_absence'),
    (1, 'counterfactual_swap'),
]


def draw_text(draw, xy, text, font, fill):
    draw.text(xy, text, font=font, fill=fill)


def render(frame, mask, query, tag, present, e_pred, font):
    """Overlay mask + query + badges on a frame; returns BGR ndarray."""
    img = frame.convert('RGB')
    if mask is not None and mask.any():
        over = np.array(img).copy()
        m = mask.astype(bool)
        over[m] = (over[m] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
        img = Image.fromarray(over)
    d = ImageDraw.Draw(img)
    # top bar: tag + query
    d.rectangle([0, 0, img.width, 34], fill=(20, 20, 20))
    draw_text(d, (6, 4), f'{tag} | {query[:70]}', font, (255, 255, 255))
    # bottom badges
    col = (0, 200, 0) if present else (220, 40, 40)
    d.rectangle([0, img.height - 26, 150, img.height], fill=(20, 20, 20))
    draw_text(d, (6, img.height - 22),
              f'GT: {"PRESENT" if present else "ABSENT"} | e_t={"1" if e_pred else "0"}',
              font, col)
    return np.array(img)[:, :, ::-1]  # RGB -> BGR for cv2


def run_case(model, tok, proc, c, use_temporal):
    # before: raw SAM2 propagation (gate disabled) -> hallucination through absence
    # after : temporal GRU gate -> clean stop
    model.temporal_gate_enabled = use_temporal
    frames = [Image.open(os.path.join(JPEGROOT, c['video_id'], f + '.jpg')).convert('RGB')
              for f in c['frames']]
    text = f'<image>\n Please segment {c["query"]} in this video.'
    with torch.no_grad():
        out = model.predict_forward(video=frames, text=text,
                                    tokenizer=tok, processor=proc)
    pred = out['prediction_masks'][0] if out['prediction_masks'] else None
    n = len(frames)
    masks = [pred[t] for t in range(min(n, len(pred)))] if pred is not None else []
    e_pred = [bool(masks[t].any()) if t < len(masks) else False for t in range(n)]
    return frames, masks, e_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/9950backfile/chenjiahui/evo_artifacts/data/examples')
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--titles', type=str,
                    default='64:clean_stop,184:reappear,18:identity,0:global_absent,1:counterfactual')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # parse curated titles
    curated = []
    for pair in args.titles.split(','):
        i, t = pair.split(':')
        curated.append((int(i), t))

    manifest = json.load(open(MANIFEST))
    cases = manifest['cases']

    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    model._temporal_ckpt = model.temporal_existence_head
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, title in curated:
        c = cases[idx]
        print(f'[{title}] vid={c["video_id"]} cat={c["category"]} query={c["query"][:60]}', flush=True)
        frames_b, masks_b, e_b = run_case(model, tok, proc, c, use_temporal=False)
        frames_a, masks_a, e_a = run_case(model, tok, proc, c, use_temporal=True)
        exp = c['presence']
        writer = None
        n = len(frames_a)
        for t in range(n):
            fb = render(frames_b[t], masks_b[t] if t < len(masks_b) else None,
                        c['query'], 'BEFORE (no gate)', exp[t], e_b[t], font)
            fa = render(frames_a[t], masks_a[t] if t < len(masks_a) else None,
                        c['query'], 'AFTER (+temporal head)', exp[t], e_a[t], font)
            H = max(fb.shape[0], fa.shape[0])
            side = np.concatenate([fb, fa], axis=1)
            if writer is None:
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(
                    os.path.join(args.out, f'{title}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            writer.write(side)
        if writer:
            writer.release()
        print(f'  -> {args.out}/{title}.mp4 (n={n})', flush=True)


if __name__ == '__main__':
    main()

"""Run several EvoSeg model versions on the same curated faithfulness cases and
save per-model videos + per-frame presence data for the showcase webpage.

Each model runs in its natural configuration:
  * Sa2VA-4B          : no existence gate -> SAM2 propagates through absence
  * VideoFaithful-4B  : image-level refusal only, video propagation untouched
  * TEG-4B (MLP)      : per-frame MLP e_t gate
  * TEG-4B + GRU      : temporal GRU e_t gate (after temporal_existence_head.pt
                        is placed in the TEG model dir)

Output layout (per curated case <title>):
  <out>/<title>/<label>.mp4      single-pane video with mask overlay
  <out>/<title>/<label>.json     {presence:[...], area:[...], text}
  <out>/<title>/meta.json        {query, category, video_id, n_frames,
                                   expected:[...], models:[labels]}

Usage:
  CUDA_VISIBLE_DEVICES=0 python make_model_comparison_videos.py \
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

MODELS = [
    ('Sa2VA-4B', '/9950backfile/chenjiahui/evo_artifacts/models/Sa2VA-Qwen3-VL-4B', 'none'),
    ('VideoFaithful-4B', '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-VideoFaithful', 'none'),
    ('TEG-4B (MLP)', '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG', 'mlp'),
    ('TEG-4B+GRU', '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG', 'gru'),
]
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/valid/JPEGImages')
MANIFEST = ('/9950backfile/chenjiahui/evo_artifacts/datasets/'
            'ref_youtube_vos/faithfulness_valid.json')
CURATED = [(64, 'clean_stop'), (184, 'reappear'), (18, 'identity'),
           (0, 'global_absent'), (1, 'counterfactual')]


def load_frames(c):
    vid = c['video_id']
    return [Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
            for f in c['frames']]


def run_model(model, tok, proc, c, mode):
    if mode == 'mlp':
        model.temporal_existence_head = None          # force MLP fallback
    if mode == 'gru' and getattr(model, 'temporal_existence_head', None) is None:
        raise RuntimeError('GRU head not present in model dir')
    frames = load_frames(c)
    text = f'<image>\n Please segment {c["query"]} in this video.'
    with torch.no_grad():
        out = model.predict_forward(video=frames, text=text,
                                    tokenizer=tok, processor=proc)
    pred = out['prediction_masks'][0] if out['prediction_masks'] else None
    n = len(frames)
    masks = [pred[t] for t in range(min(n, len(pred)))] if pred is not None else []
    presence = [bool(masks[t].any()) if t < len(masks) else False for t in range(n)]
    area = [float(masks[t].mean()) if t < len(masks) else 0.0 for t in range(n)]
    return frames, masks, presence, area, out['prediction']


def render(frame, mask, query, label, present, font):
    img = frame.convert('RGB')
    if mask is not None and mask.any():
        over = np.array(img).copy()
        m = mask.astype(bool)
        over[m] = (over[m] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
        img = Image.fromarray(over)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(20, 20, 20))
    d.text((6, 4), f'{label} | {query[:70]}', font=font, fill=(255, 255, 255))
    col = (0, 200, 0) if present else (220, 40, 40)
    d.rectangle([0, img.height - 26, 150, img.height], fill=(20, 20, 20))
    d.text((6, img.height - 22), f'GT: {"PRESENT" if present else "ABSENT"}',
           font=font, fill=col)
    return np.array(img)[:, :, ::-1]  # RGB -> BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/9950backfile/chenjiahui/evo_artifacts/data/examples')
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--cases', type=str,
                    default='64:clean_stop,184:reappear,18:identity,0:global_absent,1:counterfactual')
    ap.add_argument('--models', type=str, default='all')
    args = ap.parse_args()

    curated = [tuple(p.split(':')) for p in args.cases.split(',')]
    curated = [(int(i), t) for i, t in curated]
    models = MODELS if args.models == 'all' else [
        m for m in MODELS if m[0] in args.models.split(',')]

    manifest = json.load(open(MANIFEST))
    cases = manifest['cases']

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except Exception:
        font = ImageFont.load_default()

    # load each model once, reuse across cases
    for label, path, mode in models:
        if mode == 'gru' and not os.path.exists(
                os.path.join(path, 'temporal_existence_head.pt')):
            print(f'[skip] {label}: temporal head not trained yet', flush=True)
            continue
        print(f'[load] {label} ({path})', flush=True)
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=True, trust_remote_code=True).eval().cuda()
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)

        for idx, title in curated:
            c = cases[idx]
            cdir = os.path.join(args.out, title)
            os.makedirs(cdir, exist_ok=True)
            frames, masks, presence, area, pred_text = run_model(model, tok, proc, c, mode)
            # video
            writer = None
            n = len(frames)
            for t in range(n):
                fr = render(frames[t], masks[t] if t < len(masks) else None,
                            c['query'], label, c['presence'][t], font)
                if writer is None:
                    h, w = fr.shape[:2]
                    writer = cv2.VideoWriter(
                        os.path.join(cdir, f'{label}.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
                writer.write(fr)
            if writer:
                writer.release()
            with open(os.path.join(cdir, f'{label}.json'), 'w') as f:
                json.dump({'presence': presence, 'area': area,
                           'text': pred_text[:200]}, f)
            print(f'  [{title}] {label}: {"".join("1" if x else "0" for x in presence)} '
                  f'(GT {"".join("1" if x else "0" for x in c["presence"][:n])})', flush=True)
        del model
        torch.cuda.empty_cache()

    # write meta.json per case
    for idx, title in curated:
        c = cases[idx]
        cdir = os.path.join(args.out, title)
        labels = [os.path.splitext(f)[0] for f in os.listdir(cdir)
                  if f.endswith('.mp4')]
        meta = {'title': title, 'category': c['category'],
                'video_id': c['video_id'], 'query': c['query'],
                'n_frames': len(c['presence']), 'expected': c['presence'],
                'models': labels}
        with open(os.path.join(cdir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

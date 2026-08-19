"""Generate the visual case panels for EvoSeg Figure 1.

Runs three checkpoints (Sa2VA base / Faithful / TEG) on one image false-premise
case and one video temporal-absence case, saving RGB overlays of the predicted
masks. The video uses frame-wise image-mode inference so the per-frame existence
predicate e_t is what decides whether a mask is emitted (this is the mechanism
that yields the 'clean stop'; the TEG gate is its efficient approximation).

Outputs (written to --out-dir):
  image_case_<model>.png          : image + overlay + query/answer caption
  video_case_<model>_f<idx>.png   : sampled frames with overlay
  video_case_<model>_presence.png : presence timeline strip
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

IMAGE_PATH = ('/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014/'
              'COCO_train2014_000000274667.jpg')
VIDEO_DIR = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
             'extracted/valid/JPEGImages/0788b4033d')
VIDEO_FRAMES = ['00150', '00155', '00160', '00165', '00170', '00175',
                '00180', '00185', '00190', '00195', '00200', '00205',
                '00210', '00215', '00220', '00225', '00230', '00235',
                '00240']

MODEL_PATHS = {
    'sa2va': '/9950backfile/chenjiahui/evo_artifacts/models/Sa2VA-Qwen3-VL-4B',
    'faithful': ('/9950backfile/chenjiahui/evo_artifacts/models/'
                 'EvoSeg-Qwen3-VL-4B-Faithful'),
    'teg': ('/9950backfile/chenjiahui/evo_artifacts/models/'
            'EvoSeg-Qwen3-VL-4B-TEG'),
}

IMAGE_QUERY = 'the red jacket'
VIDEO_QUERY = 'a man walkng in an all black outfit'


def overlay(img, mask, color=(220, 40, 40), alpha=0.55):
    """Return a copy of `img` with `mask` (HxW bool) tinted by `color`."""
    img = img.convert('RGB')
    arr = np.asarray(img).copy()
    m = np.asarray(mask, dtype=bool)
    if m.shape != arr.shape[:2]:
        m = Image.fromarray(m).resize((arr.shape[1], arr.shape[0]))
        m = np.asarray(m, dtype=bool)
    arr[m] = (alpha * np.array(color) + (1 - alpha) * arr[m]).astype(np.uint8)
    return Image.fromarray(arr)


def caption(img, text, font_size=24):
    """Add a text caption bar under the image."""
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'
                                  'DejaVuSans-Bold.ttf', font_size)
    except Exception:
        pass
    w, h = img.size
    pad = font_size + 16
    canvas = Image.new('RGB', (w, h + pad), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, h + 6), text, fill=(20, 20, 20), font=font)
    return canvas


def run_image(model, tokenizer, processor, img_path, query):
    img = Image.open(img_path).convert('RGB')
    text = f'<image>\n Please segment {query} in this image.'
    with torch.no_grad():
        out = model.predict_forward(image=img, text=text,
                                    tokenizer=tokenizer, processor=processor)
    pred = out['prediction']
    masks = out['prediction_masks']
    m = None
    if len(masks) > 0 and masks[0].size:
        m = masks[0][0] > 0  # [1,H,W] -> [H,W]
    return img, pred, m


def run_video_frame_wise(model, tokenizer, processor, query):
    """Per-frame image-mode inference over the video (e_t per frame)."""
    text = f'<image>\n Please segment {query} in this video.'
    frames, pres, masks = [], [], []
    for f in VIDEO_FRAMES:
        img = Image.open(os.path.join(VIDEO_DIR, f + '.jpg')).convert('RGB')
        frames.append(img)
        with torch.no_grad():
            out = model.predict_forward(image=img, text=text,
                                        tokenizer=tokenizer,
                                        processor=processor)
        m = None
        if len(out['prediction_masks']) > 0 and out['prediction_masks'][0].size:
            m = out['prediction_masks'][0][0] > 0
        masks.append(m)
        pres.append(bool(m is not None and m.sum() > 0))
    return frames, pres, masks


def run_video_one_shot(model, tokenizer, processor, query):
    """One-shot video-mode inference: single [SEG] + SAM2 propagation.

    This is Sa2VA's native RVOS behaviour: once the LLM emits [SEG], SAM2
    propagates the mask over the whole clip, so the mask persists after the
    referent leaves the frame (temporal hallucination).
    """
    text = f'<image>\n Please segment {query} in this video.'
    frames = [Image.open(os.path.join(VIDEO_DIR, f + '.jpg')).convert('RGB')
              for f in VIDEO_FRAMES]
    with torch.no_grad():
        out = model.predict_forward(video=frames, text=text,
                                    tokenizer=tokenizer, processor=processor)
    masks_all = out['prediction_masks']
    masks = []
    if len(masks_all) > 0 and masks_all[0].size:
        m = masks_all[0]  # [T,H,W] bool
        for t in range(len(frames)):
            masks.append(m[t] > 0 if m.ndim == 3 else m[0] > 0)
    else:
        masks = [None] * len(frames)
    pres = [bool(m is not None and m.sum() > 0) for m in masks]
    return frames, pres, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['sa2va', 'faithful', 'teg'])
    ap.add_argument('--video-mode-models', nargs='*', default=['sa2va'],
                    help='models to run video-mode (one-shot propagation) for')
    ap.add_argument('--out-dir',
                    default='/9950backfile/chenjiahui/evo_artifacts/figures')
    ap.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.models:
        path = MODEL_PATHS[name]
        print(f'[load {name}] {path}', flush=True)
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=True, trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

        # ---- image case ----
        img, pred, m = run_image(model, tokenizer, processor, IMAGE_PATH,
                                 IMAGE_QUERY)
        viz = overlay(img, m) if m is not None else img
        tag = f'Sa2VA-4B' if name == 'sa2va' else f'EvoSeg-{name.upper()}-4B'
        has_mask = 'HALLUCINATED MASK' if m is not None else 'ABSTAINS (no mask)'
        viz = caption(viz, f'{tag} | "{IMAGE_QUERY}" | {has_mask}')
        viz.save(os.path.join(args.out_dir, f'image_case_{name}.png'))
        print(f'  image: {pred[:60]!r} mask={m is not None}', flush=True)

        # ---- video case (frame-wise) ----
        if name != 'faithful':
            if name in args.video_mode_models:
                frames, pres, masks = run_video_one_shot(
                    model, tokenizer, processor, VIDEO_QUERY)
                mode = 'video-mode'
                suffix = name + '_video'
            else:
                frames, pres, masks = run_video_frame_wise(
                    model, tokenizer, processor, VIDEO_QUERY)
                mode = 'frame-wise'
                suffix = name + '_fw'
            n = len(frames)
            print(f'  video [{mode}] presence: '
                  f'{"".join("1" if p else "0" for p in pres)}', flush=True)
            # sample frames: first (present), around disappearance, late absent
            idxs = sorted(set([0, 4, 11, 12, 13, 15, 18]))
            idxs = [i for i in idxs if i < n]
            for i in idxs:
                ov = overlay(frames[i], masks[i]) if masks[i] is not None \
                    else frames[i]
                ov = caption(ov, f'{tag} | frame {VIDEO_FRAMES[i]} | '
                                 f'presence={"YES" if pres[i] else "no"}')
                ov.save(os.path.join(args.out_dir,
                                     f'video_case_{suffix}_f{i}.png'))
            # presence timeline strip
            strip = Image.new('RGB', (n * 14 + 4, 34), (255, 255, 255))
            d = ImageDraw.Draw(strip)
            for i, p in enumerate(pres):
                d.rectangle([4 + i * 14, 6, 4 + i * 14 + 10, 26],
                            fill=(60, 160, 60) if p else (200, 200, 200))
            strip.save(os.path.join(args.out_dir,
                                    f'video_case_{suffix}_presence.png'))
        del model
        torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()

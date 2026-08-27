"""Render GT (Ref-YT-VOS annotations) mask videos for showcase cases.

For each curated case: read per-frame annotation PNG (pixel value = object id),
extract the target obj_id mask, render a blue-overlay GT video + presence json,
and append 'GT' to meta.json models so the webpage shows it as the reference row.
"""
import json
import os
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

ANNROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/Annotations'
JPEGROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'
EXAMPLES = '/9950backfile/chenjiahui/evo_artifacts/data/examples'
MANIFEST = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_valid.json'

CASES = [  # (idx, title) -- demo cases (Faithful+e_t perfect on boundary)
    (64, 'man_black'), (1224, 'tissue'), (1616, 'mouse'), (1787, 'ball'),
    (583, 'surfboard_man'), (1261, 'toilet'), (601, 'surfboard'), (597, 'surfboard_identity'),
]

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except Exception:
    font = ImageFont.load_default()


def render(frame, mask, label, present):
    img = frame.convert('RGB')
    if mask is not None and mask.any():
        over = np.array(img).copy()
        m = mask.astype(bool)
        over[m] = (over[m] * 0.4 + np.array([30, 144, 255]) * 0.6).astype(np.uint8)  # blue GT
        img = Image.fromarray(over)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(20, 20, 20))
    d.text((6, 4), label, font=font, fill=(255, 255, 255))
    col = (0, 200, 0) if present else (220, 40, 40)
    d.rectangle([0, img.height - 26, 150, img.height], fill=(20, 20, 20))
    d.text((6, img.height - 22), f'GT: {"PRESENT" if present else "ABSENT"}',
           font=font, fill=col)
    return np.array(img)


def main():
    man = json.load(open(MANIFEST))
    cases = man['cases']
    for idx, title in CASES:
        c = cases[idx]
        vid = c['video_id']
        obj_id = int(c['obj_id'])
        cdir = os.path.join(EXAMPLES, title)
        os.makedirs(cdir, exist_ok=True)
        frames = []
        masks = []
        presence = []
        for f in c['frames']:
            fr = Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
            ann = np.array(Image.open(os.path.join(ANNROOT, vid, str(obj_id), f + '.png')))
            m = (ann > 0)
            frames.append(fr)
            masks.append(m)
            presence.append(bool(m.sum()))
        # video
        vpath = os.path.join(cdir, 'GT (标注).mp4')
        writer = imageio.get_writer(vpath, fps=4, codec='libx264', quality=8, pixelformat='yuv420p')
        for t in range(len(frames)):
            writer.append_data(render(frames[t], masks[t], f'GT (Ref-YT-VOS) obj={obj_id}',
                                      presence[t]))
        writer.close()
        with open(os.path.join(cdir, 'GT (标注).json'), 'w') as f:
            json.dump({'presence': presence, 'area': [float(m.mean()) for m in masks],
                       'text': f'GT obj_id={obj_id}'}, f)
        # append GT to meta.json models
        meta_p = os.path.join(cdir, 'meta.json')
        meta = json.load(open(meta_p))
        if 'GT (标注)' not in meta['models']:
            meta['models'] = meta['models'] + ['GT (标注)']
        json.dump(meta, open(meta_p, 'w'), ensure_ascii=False, indent=2)
        print(f'[{title}] GT obj={obj_id}: {"".join("1" if x else "0" for x in presence)} '
              f'(expected {"".join("1" if x else "0" for x in c["presence"])})', flush=True)


if __name__ == '__main__':
    main()

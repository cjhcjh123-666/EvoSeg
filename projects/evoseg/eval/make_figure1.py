"""Compose EvoSeg Figure 1 (opening figure).

Panels:
  (a) headline bar chart: absent-target hallucination rate ↓ and
      segmentation quality → (image and video)
  (b) image false-premise case: Sa2VA hallucinates a mask vs EvoSeg abstains
  (c) video temporal-absence case: Sa2VA propagates the mask after the target
      leaves vs EvoSeg (frame-wise e_t) stops cleanly

Run after make_figure_cases.py produced the case PNGs in --cases-dir.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def panel_a(ax):
    """Hallucination rate (left y) + seg quality (right y, dotted)."""
    models = ['Sa2VA-4B', 'EvoSeg-4B']
    img_hall = [100.0, 14.7]     # 8905 absent queries
    vid_hall = [91.3, 6.1]       # 1986 video cases (TEG)
    img_refcoco = [81.95, 82.22]
    img_grefcoco = [29.8, 69.8]

    x = [0, 1]
    w = 0.35
    c_sa = '#c0392b'
    c_ev = '#2980b9'
    ax.bar([i - w / 2 for i in x], img_hall, w, color=[c_sa, c_ev],
           label='Image absent halluc. (%)', alpha=0.92)
    ax.bar([i + w / 2 for i in x], vid_hall, w, color=[c_sa, c_ev],
           hatch='//', label='Video absent halluc. (%)', alpha=0.92)
    for i in x:
        ax.text(i - w / 2, img_hall[i] + 2, f'{img_hall[i]:.1f}',
                ha='center', fontsize=13, fontweight='bold', color=c_sa)
        ax.text(i + w / 2, vid_hall[i] + 2, f'{vid_hall[i]:.1f}',
                ha='center', fontsize=13, fontweight='bold', color=c_ev)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel('Absent-target hallucination rate (%)', fontsize=12)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('(a)  Segmentation accuracy ≠ grounding faithfulness',
                 fontsize=13, loc='left', weight='bold')

    ax2 = ax.twinx()
    ax2.plot(x, img_refcoco, 'o--', color='#27ae60', lw=2, ms=8,
             label='RefCOCO cIoU (img)')
    ax2.plot(x, img_grefcoco, 's--', color='#8e44ad', lw=2, ms=8,
             label='gRefCOCO cIoU (img)')
    for i in x:
        ax2.annotate(f'{img_refcoco[i]:.1f}', (i, img_refcoco[i]),
                     textcoords='offset points', xytext=(8, 8),
                     fontsize=10, color='#27ae60')
        ax2.annotate(f'{img_grefcoco[i]:.1f}', (i, img_grefcoco[i]),
                     textcoords='offset points', xytext=(8, -14),
                     fontsize=10, color='#8e44ad')
    ax2.set_ylabel('Referring segmentation cIoU ↑', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='center left', fontsize=10, framealpha=0.9)


def panel_b(ax, cases_dir):
    from PIL import ImageDraw, ImageFont
    sa = Image.open(os.path.join(cases_dir, 'image_case_sa2va.png'))
    fa = Image.open(os.path.join(cases_dir, 'image_case_faithful.png'))

    def label(img, color):
        lab = Image.new('RGB', (120, img.height), color)
        d = ImageDraw.Draw(lab)
        try:
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'
                                     'DejaVuSans-Bold.ttf', 22)
        except Exception:
            fnt = ImageFont.load_default()
        d.text((6, 20), 'Sa2VA-4B', fill=(120, 20, 20), font=fnt)
        d.text((6, 50), '[SEG] +', fill=(120, 20, 20), font=fnt)
        d.text((6, 76), 'mask', fill=(120, 20, 20), font=fnt)
        return lab

    lab_sa = label(sa, (252, 228, 214))
    lab_fa = label(fa, (221, 235, 247))
    d2 = ImageDraw.Draw(lab_fa)
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'
                                 'DejaVuSans-Bold.ttf', 22)
    except Exception:
        fnt = ImageFont.load_default()
    d2.text((6, 20), 'EvoSeg-4B', fill=(10, 40, 90), font=fnt)
    d2.text((6, 50), 'ABSTAINS', fill=(10, 40, 90), font=fnt)
    d2.text((6, 76), 'no mask', fill=(10, 40, 90), font=fnt)

    w = max(lab_sa.width, lab_fa.width) + max(sa.width, fa.width)
    h = sa.height + fa.height
    canvas = Image.new('RGB', (w, h), 'white')
    canvas.paste(lab_sa, (0, 0))
    canvas.paste(sa, (lab_sa.width, 0))
    canvas.paste(lab_fa, (0, sa.height))
    canvas.paste(fa, (lab_fa.width, sa.height))
    ax.imshow(canvas)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(b)  Image false premise:  “the red jacket” (absent)',
                 fontsize=13, loc='left', weight='bold')


def panel_c(ax, cases_dir):
    """Video timeline: presence strips + sampled frames."""
    from PIL import Image
    sa_strip = Image.open(os.path.join(cases_dir,
                                       'video_case_sa2va_video_presence.png'))
    te_strip = Image.open(os.path.join(cases_dir,
                                       'video_case_teg_fw_presence.png'))
    cols = (0, 11, 13, 18)
    sa_frames = [Image.open(os.path.join(cases_dir,
                                         f'video_case_sa2va_video_f{i}.png'))
                 for i in cols]
    te_frames = [Image.open(os.path.join(cases_dir,
                                         f'video_case_teg_fw_f{i}.png'))
                 for i in cols]

    # label bars
    lab_sa = Image.new('RGB', (110, sa_frames[0].height), (252, 228, 214))
    lab_te = Image.new('RGB', (110, te_frames[0].height), (221, 235, 247))
    from PIL import ImageDraw, ImageFont
    d1 = ImageDraw.Draw(lab_sa)
    d2 = ImageDraw.Draw(lab_te)
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'
                                 'DejaVuSans-Bold.ttf', 22)
    except Exception:
        fnt = ImageFont.load_default()
    d1.text((6, 20), 'Sa2VA-4B', fill=(120, 20, 20), font=fnt)
    d1.text((6, 46), 'one-shot', fill=(120, 20, 20), font=fnt)
    d1.text((6, 70), '[SEG]+SAM2', fill=(120, 20, 20), font=fnt)
    d2.text((6, 20), 'EvoSeg-4B', fill=(10, 40, 90), font=fnt)
    d2.text((6, 46), 'frame-wise', fill=(10, 40, 90), font=fnt)
    d2.text((6, 70), 'e_t', fill=(10, 40, 90), font=fnt)

    def hstack(lab, frames):
        fw = lab.width + sum(f.width for f in frames)
        fh = max([lab.height] + [f.height for f in frames])
        row = Image.new('RGB', (fw, fh), 'white')
        row.paste(lab, (0, 0))
        x0 = lab.width
        for f in frames:
            row.paste(f, (x0, 0))
            x0 += f.width
        return row

    row_sa = hstack(lab_sa, sa_frames)
    row_te = hstack(lab_te, te_frames)
    w = max(row_sa.width, row_te.width)
    canvas = Image.new('RGB', (w, row_sa.height + row_te.height), 'white')
    canvas.paste(row_sa, (0, 0))
    canvas.paste(row_te, (0, row_sa.height))
    ax.imshow(canvas)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(c)  Video: “a man walking in all-black outfit”\n'
                 'referent disappears at frame 12 → mask must stop',
                 fontsize=13, loc='left', weight='bold')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases-dir',
                    default='/9950backfile/chenjiahui/evo_artifacts/figures')
    ap.add_argument('--out',
                    default='/9950backfile/chenjiahui/evo_artifacts/figures/'
                            'figure1.png')
    args = ap.parse_args()

    fig = plt.figure(figsize=(19, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 1.15, 1.35])
    axa = fig.add_subplot(gs[0, 0])
    panel_a(axa)

    axb = fig.add_subplot(gs[:, 1])
    panel_b(axb, args.cases_dir)

    axc = fig.add_subplot(gs[:, 2])
    panel_c(axc, args.cases_dir)

    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()

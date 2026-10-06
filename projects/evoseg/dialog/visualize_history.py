"""Measured native GT-history vs closed-loop cases, not curated method gains."""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from .protocol import decode_annotation
from .public_data import parse_conversation
from projects.evoseg.restart.prepare_foundation import atomic_json


def overlay(image, mask, color):
    array = np.asarray(image).copy()
    array[mask] = (array[mask] * .55 + np.asarray(color) * .45).astype(np.uint8)
    return Image.fromarray(array)


def build(args):
    root = Path(args.baselines)
    candidates = []
    for dataset in ('refcoco', 'refcoco+', 'refcocog'):
        for path in sorted((root / 'predicted_history' / dataset / 'cases').glob('*.json')):
            predicted = json.loads(path.read_text())
            gt = json.loads((root / 'gt_history' / dataset / 'cases' / path.name).read_text())
            for a, b in zip(gt['rounds'], predicted['rounds']):
                if a['round'] != b['round']:
                    raise RuntimeError('round mismatch')
                if 2 <= a['round'] <= 6:
                    candidates.append({'dataset': dataset, 'index': predicted['index'],
                        'round': a['round'], 'image': predicted['image'], 'gt_history': a,
                        'predicted_history': b, 'drop': a['gIoU'] - b['gIoU']})
    chosen, seen = [], set()
    # Explicit diagnostic selection: three largest drops and three strong
    # closed-loop cases. Never present this selection as overall success rate.
    for label, sequence in [('largest_history_drop', sorted(candidates, key=lambda x: -x['drop'])),
                            ('strong_closed_loop', sorted(candidates, key=lambda x: -x['predicted_history']['gIoU']))]:
        count = 0
        for item in sequence:
            if item['image'] in seen:
                continue
            seen.add(item['image'])
            chosen.append({**item, 'selection': label})
            count += 1
            if count == 3:
                break
    if len(chosen) != 6:
        raise RuntimeError('need complete diagnostic inventory')
    annotations = {int(a['id']): a for a in json.loads(Path(args.annotations).read_text())['annotations']}
    raw = {d: json.loads((Path(args.dialogue_root) / f'mr_{d}_val.json').read_text())
           for d in ('refcoco', 'refcoco+', 'refcocog')}
    panel_w, panel_h, title_h = 300, 240, 110
    canvas = Image.new('RGB', (4 * panel_w, len(chosen) * (panel_h + title_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, item in enumerate(chosen):
        turn = parse_conversation(raw[item['dataset']][item['index']])[item['round'] - 1]
        with Image.open(item['image']) as original:
            image = original.convert('RGB')
        target = decode_annotation(annotations[turn['source_annotation_id']], image.height, image.width)
        masks = [mask_utils.decode(item[mode]['prediction_rle']).astype(bool)
                 for mode in ('gt_history', 'predicted_history')]
        y = row * (panel_h + title_h)
        title = (f'{item["selection"]} | {item["dataset"]} case {item["index"]} round {item["round"]}\n'
                 f'GT-history IoU {item["gt_history"]["gIoU"]:.3f} | '
                 f'closed-loop IoU {item["predicted_history"]["gIoU"]:.3f}\n')
        title += '\n'.join(textwrap.wrap(turn['query'], width=155)[:3])
        draw.text((5, y + 5), title, fill='black')
        panels = [image, overlay(image, target, (40, 255, 40)),
                  overlay(image, masks[0], (20, 180, 255)), overlay(image, masks[1], (255, 80, 40))]
        for column, panel in enumerate(panels):
            panel.thumbnail((panel_w, panel_h - 18))
            canvas.paste(panel, (column * panel_w, y + title_h + 18))
            draw.text((column * panel_w + 5, y + title_h),
                      ('Image', 'GT target', 'GT-history prediction', 'Closed-loop prediction')[column], fill='black')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / 'HISTORY_DIAGNOSTIC.png')
    atomic_json(output / 'VISUALIZATION_MANIFEST.json', {
        'selection': '3 largest round2-6 history drops and 3 strongest closed-loop cases, unique images',
        'selection_is_representative': False, 'method_gain_shown': False,
        'cases': [{key: value for key, value in item.items() if key not in ('gt_history', 'predicted_history')}
                  for item in chosen]})
    print(str(output / 'HISTORY_DIAGNOSTIC.png'), flush=True)


def main():
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines', default=str(base / 'results/evoseg_dialog_20261006/public_baselines'))
    p.add_argument('--annotations', default=str(base / 'datasets/coco2014/annotations/instances_train2014.json'))
    p.add_argument('--dialogue-root', default=str(base / 'datasets/SegLLM-official/conversations_folder/all_data_mix_val'))
    p.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/baseline_visualization'))
    build(p.parse_args())


if __name__ == '__main__':
    main()

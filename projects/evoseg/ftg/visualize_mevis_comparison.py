"""Create reproducible temporal mask comparison sheets for MeViS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


def decode(rle, shape=None):
    if rle is None:
        return np.zeros(shape, dtype=bool) if shape else None
    return mask_utils.decode(rle).astype(bool)


def region_j(prediction, target):
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def target_mask(mask_dictionary, annotation_ids, frame_index, shape):
    target = np.zeros(shape, dtype=bool)
    for annotation_id in annotation_ids:
        masks = mask_dictionary.get(str(annotation_id), [])
        if frame_index < len(masks) and masks[frame_index] is not None:
            target |= decode(masks[frame_index])
    return target


def expression_records(baseline, ftg, metadata, mask_dictionary):
    records = []
    for video_id in sorted(set(baseline) & set(ftg)):
        expressions = metadata[video_id]["expressions"]
        for expression_id in sorted(set(baseline[video_id]) & set(ftg[video_id])):
            base_item = baseline[video_id][expression_id]
            ftg_item = ftg[video_id][expression_id]
            info = expressions.get(expression_id)
            if info is None:
                continue
            base_masks = base_item["prediction_masks"]
            ftg_masks = ftg_item["prediction_masks"]
            count = min(len(base_masks), len(ftg_masks))
            base_js, ftg_js = [], []
            for frame_index in range(count):
                base_mask = decode(base_masks[frame_index])
                ftg_mask = decode(ftg_masks[frame_index])
                target = target_mask(
                    mask_dictionary,
                    info["anno_id"],
                    frame_index,
                    base_mask.shape,
                )
                if target.any():
                    base_js.append(region_j(base_mask, target))
                    ftg_js.append(region_j(ftg_mask, target))
            if base_js:
                records.append({
                    "video_id": video_id,
                    "expression_id": expression_id,
                    "expression": info["exp"],
                    "frames": base_item["frames"],
                    "annotation_ids": info["anno_id"],
                    "baseline_j": float(np.mean(base_js)),
                    "ftg_j": float(np.mean(ftg_js)),
                    "delta_j": float(np.mean(ftg_js) - np.mean(base_js)),
                })
    return records


def overlay(image, mask, color=(20, 220, 80), alpha=0.48):
    array = np.asarray(image.convert("RGB")).copy()
    color_array = np.empty_like(array)
    color_array[:] = color
    array[mask] = (
        array[mask].astype(np.float32) * (1 - alpha)
        + color_array[mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return Image.fromarray(array)


def fit(image, size):
    canvas = Image.new("RGB", size, "black")
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(image, ((size[0] - image.width) // 2,
                         (size[1] - image.height) // 2))
    return canvas


def render(
    record,
    baseline,
    ftg,
    mask_dictionary,
    image_root,
    output,
    baseline_label="Strong baseline",
    candidate_label="FTG",
):
    video_id = record["video_id"]
    expression_id = record["expression_id"]
    base_item = baseline[video_id][expression_id]
    ftg_item = ftg[video_id][expression_id]
    frames = record["frames"]
    count = min(len(frames), len(base_item["prediction_masks"]),
                len(ftg_item["prediction_masks"]))
    indices = np.linspace(0, count - 1, min(4, count), dtype=int).tolist()
    cell = (320, 190)
    header_height = 76
    sheet = Image.new("RGB", (cell[0] * 4, header_height + cell[1] * len(indices)), "white")
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{video_id}/{expression_id}  "
        f"J: {baseline_label} {record['baseline_j']:.3f} | "
        f"{candidate_label} {record['ftg_j']:.3f} "
        f"| delta {record['delta_j']:+.3f}"
    )
    draw.text((8, 6), title, fill="black")
    query = record["expression"]
    draw.text((8, 27), query[:190], fill="black")
    for column, label in enumerate(
        ("RGB", "Ground truth", baseline_label, candidate_label)
    ):
        draw.text((column * cell[0] + 8, 54), label, fill="black")
    for row, frame_index in enumerate(indices):
        frame_name = frames[frame_index]
        image = Image.open(image_root / video_id / f"{frame_name}.jpg").convert("RGB")
        base_mask = decode(base_item["prediction_masks"][frame_index])
        ftg_mask = decode(ftg_item["prediction_masks"][frame_index])
        target = target_mask(
            mask_dictionary,
            record["annotation_ids"],
            frame_index,
            base_mask.shape,
        )
        panels = (image, overlay(image, target), overlay(image, base_mask),
                  overlay(image, ftg_mask))
        for column, panel in enumerate(panels):
            sheet.paste(fit(panel, cell), (column * cell[0], header_height + row * cell[1]))
        draw.text((4, header_height + row * cell[1] + 4), frame_name,
                  fill="white", stroke_width=2, stroke_fill="black")
    sheet.save(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ftg", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-side", type=int, default=4)
    parser.add_argument("--baseline-label", default="Strong baseline")
    parser.add_argument("--candidate-label", default="FTG")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    ftg = json.loads(args.ftg.read_text())
    metadata = json.loads(args.meta.read_text())["videos"]
    mask_dictionary = json.loads(args.mask.read_text())
    records = expression_records(baseline, ftg, metadata, mask_dictionary)
    records.sort(key=lambda item: item["delta_j"])
    selected = records[:args.per_side] + records[-args.per_side:]
    args.output.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected):
        side = "worst" if index < args.per_side else "best"
        filename = (
            f"{index:02d}_{side}_{record['video_id']}_"
            f"{record['expression_id']}.jpg"
        )
        render(
            record,
            baseline,
            ftg,
            mask_dictionary,
            args.image_root,
            args.output / filename,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
        )
        record["file"] = filename
    (args.output / "manifest.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    print(json.dumps({"candidates": len(records), "rendered": len(selected),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

"""Render paired Long-RVOS qualitative sheets selected by metric deltas.

Cases are selected from per-expression official J/F scores rather than by hand.
Each sheet shows the same temporal snapshots for RGB, ground truth, baseline,
and candidate predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


def decode(rle):
    return mask_utils.decode(rle).astype(bool)


def overlay(image, mask, color, alpha=0.48):
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
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def keyed_rows(path):
    rows = json.loads(path.read_text())
    return {(row["video"], str(row["expression_id"])): row for row in rows}


def paired_records(baseline_rows, candidate_rows):
    records = []
    for key in sorted(set(baseline_rows) & set(candidate_rows)):
        baseline = baseline_rows[key]
        candidate = candidate_rows[key]
        baseline_jf = (baseline["j"] + baseline["f"]) / 2
        candidate_jf = (candidate["j"] + candidate["f"]) / 2
        records.append({
            "video": key[0],
            "expression_id": key[1],
            "type": baseline["type"],
            "baseline_jf": baseline_jf,
            "candidate_jf": candidate_jf,
            "delta_jf": candidate_jf - baseline_jf,
        })
    return records


def selected_extremes(records, per_side):
    selected = []
    for expression_type in ("static", "dynamic", "hybrid"):
        typed = sorted(
            (row for row in records if row["type"] == expression_type),
            key=lambda row: row["delta_jf"],
        )
        selected.extend(("worst", row) for row in typed[:per_side])
        selected.extend(("best", row) for row in typed[-per_side:][::-1])
    return selected


def target_mask(annotations, video, object_id, frame, shape):
    path = annotations / video / str(object_id) / f"{frame}.png"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    return np.asarray(Image.open(path).convert("L")) != 0


def render(record, baseline, candidate, metadata, args, output):
    video = record["video"]
    expression_id = record["expression_id"]
    base_item = baseline[video][expression_id]
    candidate_item = candidate[video][expression_id]
    video_info = metadata[video]
    expression = video_info["expressions"][expression_id]
    frames = video_info["frames"]
    count = min(
        len(frames),
        len(base_item["prediction_masks"]),
        len(candidate_item["prediction_masks"]),
    )
    indices = np.linspace(0, count - 1, min(5, count), dtype=int).tolist()
    cell = (320, 190)
    header_height = 82
    sheet = Image.new(
        "RGB", (cell[0] * 4, header_height + cell[1] * len(indices)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{record['type'].upper()}  {video}/{expression_id}  J&F: "
        f"{args.baseline_label} {record['baseline_jf']:.3f} | "
        f"{args.candidate_label} {record['candidate_jf']:.3f} | "
        f"delta {record['delta_jf']:+.3f}"
    )
    draw.text((8, 6), title, fill="black")
    draw.text((8, 28), expression["exp"][:190], fill="black")
    labels = ("RGB", "Ground truth", args.baseline_label, args.candidate_label)
    for column, label in enumerate(labels):
        draw.text((column * cell[0] + 8, 60), label, fill="black")

    for row, frame_index in enumerate(indices):
        frame = frames[frame_index]
        image = Image.open(args.image_root / video / f"{frame}.jpg").convert("RGB")
        base_mask = decode(base_item["prediction_masks"][frame_index])
        candidate_mask = decode(candidate_item["prediction_masks"][frame_index])
        target = target_mask(
            args.annotations, video, expression["obj_id"], frame, base_mask.shape
        )
        panels = (
            image,
            overlay(image, target, (40, 120, 255)),
            overlay(image, base_mask, (20, 220, 80)),
            overlay(image, candidate_mask, (255, 145, 20)),
        )
        y = header_height + row * cell[1]
        for column, panel in enumerate(panels):
            sheet.paste(fit(panel, cell), (column * cell[0], y))
        draw.text((4, y + 4), frame, fill="white", stroke_width=2,
                  stroke_fill="black")
    sheet.save(output, quality=92)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-side", type=int, default=2)
    parser.add_argument("--baseline-label", default="Identity anchor")
    parser.add_argument("--candidate-label", default="FTG")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    metadata = json.loads(args.meta.read_text())["videos"]
    records = paired_records(
        keyed_rows(args.baseline_rows), keyed_rows(args.candidate_rows)
    )
    selected = selected_extremes(records, args.per_side)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (side, record) in enumerate(selected):
        filename = (
            f"{index:02d}_{record['type']}_{side}_"
            f"{record['video']}_{record['expression_id']}.jpg"
        )
        render(
            record, baseline, candidate, metadata, args, args.output / filename
        )
        manifest.append({**record, "side": side, "file": filename})
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        "paired_expressions": len(records),
        "rendered": len(manifest),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

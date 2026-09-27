"""Create deterministic, category-balanced qualitative case panels."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


def read_terminal(roots: list[Path], filename: str, key_fields: tuple[str, ...]) -> dict:
    values = {}
    for root in roots:
        path = root / filename
        if not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    value["_root"] = str(root)
                    values[tuple(str(value[field]) for field in key_fields)] = value
    return values


def decode(value: dict) -> np.ndarray:
    rle = dict(value)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    value = np.asarray(image.convert("RGB"), dtype=np.float32)
    target = np.asarray(color, dtype=np.float32)
    value[mask] = value[mask] * 0.45 + target * 0.55
    return Image.fromarray(np.clip(value, 0, 255).astype(np.uint8))


def load_prediction(record: dict) -> dict[int, np.ndarray]:
    path = Path(record["_root"]) / record["prediction_masks_path"]
    return {int(value["frame_index"]): decode(value["rle"]) for value in json.loads(path.read_text())}


def make_panel(
    image: Image.Image,
    gt: np.ndarray,
    static: np.ndarray,
    temporal: np.ndarray,
    title: str,
) -> Image.Image:
    panels = [
        image.convert("RGB"),
        overlay(image, gt, (0, 210, 0)),
        overlay(image, static, (230, 60, 40)),
        overlay(image, temporal, (40, 110, 240)),
    ]
    labels = ["RGB", "GT", "Static-update K4", "Temporal-update K4"]
    width = 360
    resized = []
    for panel in panels:
        height = int(round(panel.height * width / panel.width))
        resized.append(panel.resize((width, height), Image.Resampling.BILINEAR))
    header_height = 90
    label_height = 24
    canvas = Image.new(
        "RGB", (width * 4, header_height + label_height + resized[0].height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, 6), "\n".join(textwrap.wrap(title, width=150)), fill="black", spacing=3)
    for index, (panel, label) in enumerate(zip(resized, labels)):
        x = index * width
        draw.text((x + 8, header_height + 4), label, fill="black")
        canvas.paste(panel, (x, header_height + label_height))
    return canvas


def run(args) -> None:
    roots = [Path(value).resolve() for value in args.input_root]
    predictions = read_terminal(
        roots, "dynamic_predictions.jsonl", ("identity", "condition")
    )
    stages = read_terminal(
        roots,
        "dynamic_stages.jsonl",
        ("identity", "condition", "stage_order"),
    )
    manifest = json.loads(Path(args.manifest).read_text())
    item_by_identity = {}
    expression_by_identity = {}
    for item in manifest["objects"]:
        for expression in item["expressions"]:
            identity = "/".join(
                map(
                    str,
                    (
                        item["dataset"],
                        item["video_id"],
                        item["object_id"],
                        expression["expression_id"],
                    ),
                )
            )
            item_by_identity[identity] = item
            expression_by_identity[identity] = expression

    by_identity = defaultdict(list)
    for (identity, condition, _order), value in stages.items():
        if condition == "temporal_update_k4":
            by_identity[identity].append(value)
    categorized = defaultdict(list)
    for identity, values in by_identity.items():
        if expression_by_identity[identity]["type"] != "dynamic":
            continue
        values.sort(key=lambda value: int(value["stage_order"]))
        correctness = [bool(value["selection_correct"]) for value in values]
        if not correctness[0] and any(correctness[1:]):
            category = "referent_correction"
            event = next(value for value in values[1:] if value["selection_correct"])
        elif correctness[0] and any(not value for value in correctness[1:]):
            category = "identity_damaged"
            event = next(value for value in values[1:] if not value["selection_correct"])
        else:
            category = "no_help"
            event = values[-1]
        categorized[category].append((identity, event))

    selected = []
    quotas = {"referent_correction": 3, "no_help": 3, "identity_damaged": 3}
    for category in ("referent_correction", "no_help", "identity_damaged"):
        selected.extend(
            (category, identity, event)
            for identity, event in sorted(categorized[category], key=lambda value: value[0])[
                : quotas[category]
            ]
        )
    if len(selected) < args.max_cases:
        already = {identity for _category, identity, _event in selected}
        remainder = sorted(
            (
                category,
                identity,
                event,
            )
            for category, values in categorized.items()
            for identity, event in values
            if identity not in already
        )
        selected.extend(remainder[: args.max_cases - len(selected)])
    selected = selected[: args.max_cases]

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_root = Path(manifest["dataset"]["image_root"])
    records = []
    for number, (category, identity, event) in enumerate(selected, start=1):
        static_record = predictions.get((identity, "static_update_k4"))
        temporal_record = predictions.get((identity, "temporal_update_k4"))
        if not static_record or not temporal_record:
            continue
        if static_record.get("status") != "success" or temporal_record.get("status") != "success":
            continue
        static_masks = load_prediction(static_record)
        temporal_masks = load_prediction(temporal_record)
        common = sorted(set(static_masks) & set(temporal_masks))
        event_index = int(event["anchor_frame_index"])
        frame_index = min(common, key=lambda value: (abs(value - event_index), value))
        item = item_by_identity[identity]
        frame_name = item["frame_names"][frame_index]
        with Image.open(image_root / item["video_id"] / f"{frame_name}.jpg") as source:
            image = source.convert("RGB").copy()
        gt_path = Path(item["evaluation_mask_paths"][0]).parent / f"{frame_name}.png"
        with Image.open(gt_path) as source:
            gt = np.asarray(source.convert("L")) > 0
        expression = expression_by_identity[identity]["text"]
        title = (
            f"{category} | {identity} | frame={frame_name} | "
            f"Static J&F={static_record['J_and_F']*100:.2f}, "
            f"Temporal J&F={temporal_record['J_and_F']*100:.2f} | {expression}"
        )
        panel = make_panel(
            image, gt, static_masks[frame_index], temporal_masks[frame_index], title
        )
        filename = f"{number:02d}_{category}_{identity.replace('/', '__')}.png"
        panel.save(output / filename)
        records.append(
            {
                "category": category,
                "identity": identity,
                "expression": expression,
                "event_anchor_frame_index": event_index,
                "shown_frame_index": frame_index,
                "shown_frame_name": frame_name,
                "static_J_and_F": static_record["J_and_F"],
                "temporal_J_and_F": temporal_record["J_and_F"],
                "file": filename,
                "selection_rule": "deterministic identity order within prespecified transition category",
            }
        )
    (output / "case_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


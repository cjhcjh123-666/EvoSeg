"""Prepare fixed Long-RVOS keys for VIRST and convert PNG predictions to RLE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def prepare(
    source_root: Path,
    selected_rows: Path,
    output_root: Path,
) -> dict:
    """Build a GT-free MeViS-test view for exactly the selected expressions."""
    source_meta = json.loads(
        (source_root / "meta_expressions.json").read_text()
    )["videos"]
    rows = json.loads(selected_rows.read_text())
    target = output_root / "mevis" / "valid"
    jpeg_target = target / "JPEGImages"
    jpeg_target.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    mapping = []
    seen = set()
    for row in rows:
        video = str(row["video"])
        expression_id = str(row["expression_id"])
        key = (video, expression_id)
        if key in seen:
            raise ValueError(f"duplicate selected expression: {video}/{expression_id}")
        seen.add(key)
        source_video = source_meta[video]
        expression = source_video["expressions"][expression_id]
        output_expression = f"expression-{expression_id}"
        video_record = selected.setdefault(
            video,
            {"frames": source_video["frames"], "expressions": {}},
        )
        video_record["expressions"][output_expression] = {
            "exp": expression["exp"],
        }
        source_images = (source_root / "JPEGImages" / video).resolve()
        link = jpeg_target / video
        if link.is_symlink() or link.exists():
            if link.resolve() != source_images:
                raise RuntimeError(f"unexpected existing video link: {link}")
        else:
            os.symlink(source_images, link, target_is_directory=True)
        mapping.append({
            "video": video,
            "expression_id": expression_id,
            "output_expression": output_expression,
            "type": expression["type"],
            "object_id": int(expression["obj_id"]),
            "frames": source_video["frames"],
            "expression": expression["exp"],
        })
    (target / "meta_expressions.json").write_text(
        json.dumps({"videos": selected}, ensure_ascii=False, indent=2) + "\n"
    )
    (output_root / "expression_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    )
    audit = {
        "source_root": str(source_root.resolve()),
        "selected_rows": str(selected_rows.resolve()),
        "videos": len(selected),
        "expressions": len(mapping),
        "ground_truth_available_to_model": False,
    }
    (output_root / "prepare_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return audit


def encode_prediction(path: Path) -> dict:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def convert(mapping_path: Path, prediction_root: Path, output: Path) -> dict:
    """Convert VIRST's per-frame PNG layout to the shared scorer's RLE JSON."""
    mapping = json.loads(mapping_path.read_text())
    results: dict[str, dict] = {}
    missing = []
    for item in mapping:
        directory = (
            prediction_root / item["video"] / item["output_expression"]
        )
        predictions = []
        for frame in item["frames"]:
            path = directory / f"{frame}.png"
            if not path.is_file():
                missing.append(str(path))
                continue
            predictions.append(encode_prediction(path))
        if len(predictions) != len(item["frames"]):
            continue
        results.setdefault(item["video"], {})[item["expression_id"]] = {
            "video_id": item["video"],
            "exp_id": item["expression_id"],
            "exp": item["expression"],
            "frames": item["frames"],
            "prediction_masks": predictions,
        }
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} VIRST prediction masks; first={missing[0]}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results) + "\n")
    return {
        "expressions": sum(len(value) for value in results.values()),
        "videos": len(results),
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--selected-rows", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--mapping", type=Path, required=True)
    convert_parser.add_argument("--prediction-root", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.selected_rows, args.output_root)
    else:
        result = convert(args.mapping, args.prediction_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

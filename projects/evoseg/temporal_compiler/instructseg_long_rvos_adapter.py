"""Prepare and score Long-RVOS with InstructSeg's official RefYoutube path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def expression_key(item: dict, expression: dict) -> str:
    return "/".join(
        [
            item["dataset"],
            item["video_id"],
            str(item["object_id"]),
            str(expression["expression_id"]),
            "native",
        ]
    )


def prepare(manifest: dict, max_objects: int | None) -> tuple[dict, list[dict]]:
    objects = manifest["objects"][:max_objects] if max_objects else manifest["objects"]
    image_root = Path(manifest["dataset"]["image_root"])
    videos = []
    mapping = []
    numeric_id = 0
    for item in objects:
        first_image = image_root / item["video_id"] / f"{item['frame_names'][0]}.jpg"
        with Image.open(first_image) as image:
            width, height = image.size
        for expression in item["expressions"]:
            output_video = "__".join(
                [item["video_id"], str(item["object_id"]), str(expression["expression_id"])]
            )
            output_expression = "0"
            videos.append(
                {
                    "id": numeric_id,
                    "video": output_video,
                    "exp_id": output_expression,
                    "file_names": [
                        f"{item['video_id']}/{name}.jpg" for name in item["frame_names"]
                    ],
                    "height": height,
                    "width": width,
                    "length": len(item["frame_names"]),
                    "expressions": [expression["text"]],
                }
            )
            mapping.append(
                {
                    "key": expression_key(item, expression),
                    "dataset": item["dataset"],
                    "video_id": item["video_id"],
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "output_video": output_video,
                    "output_expression": output_expression,
                    "model_input_frame_indices": list(range(len(item["frame_names"]))),
                    "model_input_frame_names": item["frame_names"],
                    "evaluation_frame_indices": item["evaluation_frame_indices"],
                    "evaluation_frame_names": item["evaluation_frame_names"],
                    "evaluation_mask_paths": item["evaluation_mask_paths"],
                    "gt_available_to_model": False,
                    "native_reference_frame_num": 4,
                }
            )
            numeric_id += 1
    return {"videos": videos}, mapping


def evaluate_entry(entry: dict, annotation_root: Path) -> dict:
    prediction_dir = annotation_root / entry["output_video"] / entry["output_expression"]
    j_values = []
    f_values = []
    missing = []
    for frame_name, gt_path_value in zip(
        entry["evaluation_frame_names"], entry["evaluation_mask_paths"]
    ):
        prediction_path = prediction_dir / f"{frame_name}.png"
        if not prediction_path.is_file():
            missing.append(str(prediction_path))
            continue
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image.convert("L")) > 0
        with Image.open(gt_path_value) as image:
            ground_truth = np.asarray(image.convert("L")) > 0
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"mask shape mismatch for {entry['key']}: "
                f"{prediction.shape} != {ground_truth.shape}"
            )
        j_values.append(float(db_eval_iou(ground_truth, prediction)))
        f_values.append(float(db_eval_boundary(ground_truth, prediction)))
    base = {
        **{key: entry[key] for key in (
            "key", "dataset", "video_id", "object_id", "expression_id",
            "description_type", "expression",
        )},
        "frame_budget": "native_current_plus_4_references",
        "model_input_frame_indices": entry["model_input_frame_indices"],
        "evaluation_frame_indices": entry["evaluation_frame_indices"],
        "native_reference_frame_num": entry["native_reference_frame_num"],
        "gt_available_to_model": False,
        "missing_prediction_paths": missing,
    }
    if missing:
        return {**base, "status": "failed_missing_prediction"}
    j_score = float(np.mean(j_values))
    f_score = float(np.mean(f_values))
    return {
        **base,
        "status": "success",
        "J": j_score,
        "F": f_score,
        "J_and_F": (j_score + f_score) / 2,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def run_prepare(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    input_json, mapping = prepare(manifest, args.max_objects)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(input_json, indent=2, ensure_ascii=False) + "\n"
    )
    Path(args.mapping_json).write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
    )


def run_evaluate(args) -> None:
    mapping = json.loads(Path(args.mapping_json).read_text())
    rows = [evaluate_entry(entry, Path(args.annotation_root)) for entry in mapping]
    write_jsonl(Path(args.output_predictions), rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--output-json", required=True)
    prepare_parser.add_argument("--mapping-json", required=True)
    prepare_parser.add_argument("--max-objects", type=int)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--mapping-json", required=True)
    evaluate_parser.add_argument("--annotation-root", required=True)
    evaluate_parser.add_argument("--output-predictions", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        run_prepare(arguments)
    else:
        run_evaluate(arguments)

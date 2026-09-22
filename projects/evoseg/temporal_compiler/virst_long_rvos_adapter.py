"""Prepare a GT-free MeViS-test view of Long-RVOS and score VIRST outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def expression_key(item: dict, expression: dict) -> str:
    return "/".join(
        [
            item["dataset"], item["video_id"], str(item["object_id"]),
            str(expression["expression_id"]), "native",
        ]
    )


def select_objects(
    objects: list[dict],
    max_objects: int | None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"invalid shard {shard_index} of {num_shards}; expected 0 <= index < count"
        )
    selected = objects[:max_objects] if max_objects else objects
    return [item for index, item in enumerate(selected) if index % num_shards == shard_index]


def prepare(
    manifest: dict,
    dataset_root: Path,
    max_objects: int | None,
    shard_index: int = 0,
    num_shards: int = 1,
    max_expressions_per_object: int | None = None,
) -> list[dict]:
    objects = select_objects(
        manifest["objects"], max_objects, shard_index=shard_index, num_shards=num_shards
    )
    image_root = Path(manifest["dataset"]["image_root"]).resolve()
    target = dataset_root / "mevis" / "valid"
    jpeg_root = target / "JPEGImages"
    jpeg_root.mkdir(parents=True, exist_ok=True)
    videos: dict[str, dict] = {}
    mapping = []
    for item in objects:
        video_id = item["video_id"]
        source_video = image_root / video_id
        link = jpeg_root / video_id
        if link.exists() or link.is_symlink():
            if link.resolve() != source_video:
                raise RuntimeError(f"unexpected existing video link: {link}")
        else:
            os.symlink(source_video, link, target_is_directory=True)
        video = videos.setdefault(
            video_id,
            {"frames": item["frame_names"], "expressions": {}},
        )
        if video["frames"] != item["frame_names"]:
            raise AssertionError(f"frame list differs within source video {video_id}")
        expressions = item["expressions"]
        if max_expressions_per_object is not None:
            if max_expressions_per_object < 1:
                raise ValueError("max_expressions_per_object must be positive")
            expressions = expressions[:max_expressions_per_object]
        for expression in expressions:
            output_expression = (
                f"object-{item['object_id']}__expression-{expression['expression_id']}"
            )
            if output_expression in video["expressions"]:
                raise RuntimeError(f"duplicate VIRST expression id: {output_expression}")
            video["expressions"][output_expression] = {"exp": expression["text"]}
            mapping.append(
                {
                    "key": expression_key(item, expression),
                    "dataset": item["dataset"],
                    "video_id": video_id,
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "output_video": video_id,
                    "output_expression": output_expression,
                    "evaluation_frame_indices": item["evaluation_frame_indices"],
                    "evaluation_frame_names": item["evaluation_frame_names"],
                    "evaluation_mask_paths": item["evaluation_mask_paths"],
                    "evaluation_mask_present": item["evaluation_mask_present"],
                    "manifest_vlm_frame_indices_n16": item["vlm_frame_indices"]["16"],
                    "gt_available_to_model": False,
                }
            )
    (target / "meta_expressions.json").write_text(
        json.dumps({"videos": videos}, indent=2, ensure_ascii=False) + "\n"
    )
    (dataset_root / "expression_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
    )
    return mapping


def frame_audit_key(video_id: str, output_expression: str) -> str:
    return f"{video_id}/{output_expression}"


def load_frame_audit(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for output_expression in record["output_expression_ids"]:
                key = frame_audit_key(record["video_id"], output_expression)
                normalized = {**record, "output_expression": output_expression}
                normalized["superseded_frame_audit_attempts"] = (
                    records[key].get("superseded_frame_audit_attempts", 0) + 1
                    if key in records
                    else 0
                )
                records[key] = normalized
    return records


def evaluate_entry(
    entry: dict, output_root: Path, frame_audit: dict | None = None
) -> dict:
    if frame_audit is not None:
        indices = frame_audit["model_input_frame_indices"]
        if frame_audit.get("gt_available_to_model") is not False:
            raise AssertionError(f"invalid GT audit for {entry['key']}")
        if frame_audit["vlm_frame_count"] != len(indices) or frame_audit[
            "sam_frame_count"
        ] != len(indices):
            raise AssertionError(f"invalid realized frame-count audit for {entry['key']}")
    prediction_dir = output_root / entry["output_video"] / entry["output_expression"]
    j_values = []
    f_values = []
    missing = []
    present_flags = entry.get("evaluation_mask_present")
    if present_flags is None or len(present_flags) != len(
        entry["evaluation_mask_paths"]
    ):
        raise ValueError(f"missing or invalid GT-presence audit for {entry['key']}")
    for frame_name, gt_path_value, expected_present in zip(
        entry["evaluation_frame_names"],
        entry["evaluation_mask_paths"],
        present_flags,
    ):
        prediction_path = prediction_dir / f"{frame_name}.png"
        if not prediction_path.is_file():
            missing.append(str(prediction_path))
            continue
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image.convert("L")) > 0
        gt_path = Path(gt_path_value)
        if gt_path.is_file() != bool(expected_present):
            raise FileNotFoundError(
                f"GT availability differs from manifest for {entry['key']}: "
                f"expected_present={bool(expected_present)}, path={gt_path}"
            )
        if expected_present:
            with Image.open(gt_path) as image:
                ground_truth = np.asarray(image.convert("L")) > 0
        else:
            ground_truth = np.zeros(prediction.shape, dtype=bool)
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
        "frame_budget": (
            f"native_vlm_{frame_audit['vlm_frame_count']}_sam_"
            f"{frame_audit['sam_frame_count']}_outer_flex_up_to_64"
            if frame_audit is not None
            else "native_frame_audit_unavailable"
        ),
        "model_input_frame_indices": (
            frame_audit["model_input_frame_indices"]
            if frame_audit is not None
            else None
        ),
        "actual_vlm_frame_count": (
            frame_audit["vlm_frame_count"] if frame_audit is not None else None
        ),
        "actual_sam_frame_count": (
            frame_audit["sam_frame_count"] if frame_audit is not None else None
        ),
        "superseded_frame_audit_attempts": (
            frame_audit.get("superseded_frame_audit_attempts", 0)
            if frame_audit is not None
            else None
        ),
        "manifest_vlm_frame_indices_n16": entry["manifest_vlm_frame_indices_n16"],
        "evaluation_frame_indices": entry["evaluation_frame_indices"],
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


def run_prepare(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    prepare(
        manifest,
        Path(args.dataset_root).resolve(),
        args.max_objects,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_expressions_per_object=args.max_expressions_per_object,
    )


def run_evaluate(args) -> None:
    mapping = json.loads(Path(args.mapping_json).read_text())
    frame_audit = load_frame_audit(Path(args.frame_audit_jsonl))
    expected_audit_keys = {
        frame_audit_key(entry["output_video"], entry["output_expression"])
        for entry in mapping
    }
    missing_audit_keys = sorted(expected_audit_keys - set(frame_audit))
    extra_audit_keys = sorted(set(frame_audit) - expected_audit_keys)
    if missing_audit_keys or extra_audit_keys:
        raise RuntimeError(
            f"VIRST frame-audit/mapping mismatch: missing={missing_audit_keys[:5]}, "
            f"extra={extra_audit_keys[:5]}"
        )
    rows = [
        evaluate_entry(
            entry,
            Path(args.output_root),
            frame_audit[
                frame_audit_key(entry["output_video"], entry["output_expression"])
            ],
        )
        for entry in mapping
    ]
    Path(args.output_predictions).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--dataset-root", required=True)
    prepare_parser.add_argument("--max-objects", type=int)
    prepare_parser.add_argument("--shard-index", type=int, default=0)
    prepare_parser.add_argument("--num-shards", type=int, default=1)
    prepare_parser.add_argument("--max-expressions-per-object", type=int)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--mapping-json", required=True)
    evaluate_parser.add_argument("--output-root", required=True)
    evaluate_parser.add_argument("--output-predictions", required=True)
    evaluate_parser.add_argument("--frame-audit-jsonl", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        run_prepare(arguments)
    else:
        run_evaluate(arguments)

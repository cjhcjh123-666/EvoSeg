"""GroundMoRe Sequential adapter preserving its official temporal-mask protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def seconds(value: str) -> int:
    minutes, secs = value.split(":")
    return int(minutes) * 60 + int(secs)


def clip_start(video_id: str) -> int:
    value = video_id.rsplit("_", 2)[-2]
    return int(value[:2]) * 60 + int(value[2:])


def action_interval(video_id: str, start: str, end: str) -> tuple[int, int]:
    base = clip_start(video_id)
    return (seconds(start) - base) * 6, (seconds(end) - base) * 6 - 1


def object_ids(value) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def encode(mask: np.ndarray) -> dict | None:
    if not mask.any():
        return None
    value = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    value["counts"] = value["counts"].decode("ascii")
    return value


def prepare(
    source_root: Path,
    metadata_path: Path,
    output_root: Path,
    split: str,
    max_videos: int | None,
) -> dict:
    training = split == "train"
    target = output_root / "mevis" / ("train" if training else "valid")
    jpeg_root = target / "JPEGImages"
    jpeg_root.mkdir(parents=True, exist_ok=True)
    source = json.loads(metadata_path.read_text())["videos"]
    videos = {}
    mapping = []
    mask_dict = {}

    for video_id, video in source.items():
        sequential = {
            str(exp_id): item
            for exp_id, item in video["questions"].items()
            if item["q_type"].lower() == "sequential"
        }
        if not sequential:
            continue
        if max_videos is not None and len(videos) >= max_videos:
            break
        source_dir = source_root / "annotations" / video_id
        image_dir = source_dir / "images"
        mask_dir = source_dir / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"incomplete GroundMoRe clip: {video_id}")
        link = jpeg_root / video_id
        if not link.exists():
            os.symlink(image_dir, link, target_is_directory=True)
        frame_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        frame_names = [path.stem for path in frame_paths]
        expressions = {}
        for expression_id, item in sequential.items():
            ids = object_ids(item["obj_id"])
            annotation_id = f"{video_id}__expression-{expression_id}"
            start, end = action_interval(video_id, item["action_start"], item["action_end"])
            expressions[expression_id] = {
                "exp": item["question"],
                "obj_id": ids,
                "anno_id": [annotation_id],
                "q_type": item["q_type"],
            }
            if training:
                track = []
                shape = None
                for index, frame_path in enumerate(frame_paths):
                    if start <= index <= end:
                        mask_path = mask_dir / f"frame_{index:06d}.png"
                        if mask_path.is_file():
                            with Image.open(mask_path) as image:
                                raw = np.asarray(image.convert("P"))
                            shape = raw.shape
                            track.append(encode(np.isin(raw, ids)))
                        else:
                            track.append(None)
                    else:
                        track.append(None)
                mask_dict[annotation_id] = track
            mapping.append(
                {
                    "video_id": video_id,
                    "expression_id": expression_id,
                    "question": item["question"],
                    "q_type": item["q_type"],
                    "obj_ids": ids,
                    "action_start_frame": start,
                    "action_end_frame": end,
                    "frame_names": frame_names,
                }
            )
        videos[video_id] = {"frames": frame_names, "expressions": expressions}

    target.mkdir(parents=True, exist_ok=True)
    (target / "meta_expressions.json").write_text(
        json.dumps({"videos": videos}, ensure_ascii=False) + "\n"
    )
    if training:
        (target / "mask_dict.json").write_text(json.dumps(mask_dict) + "\n")
    (output_root / "groundmore_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    )
    audit = {
        "official_metadata": str(metadata_path.resolve()),
        "source_root": str(source_root.resolve()),
        "split": split,
        "videos": len(videos),
        "sequential_expressions": len(mapping),
        "fps": 6,
        "evaluation_frames": 20 if not training else None,
        "gt_outside_action_interval": "zero",
    }
    (output_root / "groundmore_adapter_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return audit


def evaluate(mapping_path: Path, output_root: Path, source_root: Path, output_csv: Path) -> dict:
    mapping = json.loads(mapping_path.read_text())
    rows = []
    for item in mapping:
        frames = item["frame_names"]
        indices = np.linspace(0, len(frames) - 1, num=20, dtype=int)
        prediction_dir = output_root / item["video_id"] / item["expression_id"]
        mask_dir = source_root / "annotations" / item["video_id"] / "masks"
        j_values, f_values = [], []
        missing = []
        for index in indices:
            name = frames[index]
            prediction_path = prediction_dir / f"{name}.png"
            if not prediction_path.is_file():
                missing.append(str(prediction_path))
                continue
            with Image.open(prediction_path) as image:
                prediction = np.asarray(image.convert("L")) > 0
            if item["action_start_frame"] <= index <= item["action_end_frame"]:
                gt_path = mask_dir / f"frame_{index:06d}.png"
                if gt_path.is_file():
                    with Image.open(gt_path) as image:
                        raw = np.asarray(image.convert("P"))
                    gt = np.isin(raw, item["obj_ids"])
                else:
                    gt = np.zeros(prediction.shape, dtype=bool)
            else:
                gt = np.zeros(prediction.shape, dtype=bool)
            j_values.append(float(db_eval_iou(gt, prediction)))
            f_values.append(float(db_eval_boundary(gt, prediction)))
        status = "success" if not missing else "failed_missing_prediction"
        j = float(np.mean(j_values)) if status == "success" else float("nan")
        f = float(np.mean(f_values)) if status == "success" else float("nan")
        rows.append({**item, "status": status, "J": j, "F": f, "J_and_F": (j + f) / 2})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id", "expression_id", "question", "q_type", "status", "J", "F", "J_and_F",
        "action_start_frame", "action_end_frame",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    success = [row for row in rows if row["status"] == "success"]
    summary = {
        "planned": len(rows),
        "success": len(success),
        "failed": len(rows) - len(success),
        "J": float(np.mean([row["J"] for row in success])) if success else None,
        "F": float(np.mean([row["F"] for row in success])) if success else None,
        "J_and_F": float(np.mean([row["J_and_F"] for row in success])) if success else None,
        "official_protocol": "20 uniform frames; instance IDs; zeros outside action interval",
    }
    (output_csv.parent / "groundmore_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--metadata", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--split", choices=["train", "test"], required=True)
    prepare_parser.add_argument("--max-videos", type=int)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--mapping", type=Path, required=True)
    evaluate_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser.add_argument("--source-root", type=Path, required=True)
    evaluate_parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.metadata, args.output_root, args.split, args.max_videos)
    else:
        result = evaluate(args.mapping, args.output_root, args.source_root, args.output_csv)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

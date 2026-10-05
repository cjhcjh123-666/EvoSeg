"""Official Long-RVOS J/F, tIoU, and vIoU from Sa2VA result JSON.

The equations and per-expression averaging follow the public Long-RVOS
``eval_{static,dynamic,hybrid}.py`` scripts.  Predictions remain in the RLE
format emitted by ``sa2va_eval_ref_vos.py`` so evaluation does not require a
second PNG export.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_seg.official_metrics import (
    db_eval_boundary,
    db_eval_iou,
)


DEFAULT_ROOT = Path(
    "/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos/valid"
)
WORKER_META = {}
WORKER_RESULTS = {}
WORKER_ANNOTATIONS = DEFAULT_ROOT / "Annotations"

# Multiprocessing already supplies parallelism. Prevent every worker from
# creating an additional OpenCV thread pool during boundary dilation.
cv2.setNumThreads(1)


def decode_bool(rle):
    return None if rle is None else mask_utils.decode(rle).astype(bool)


def evaluate_expression(task):
    video, expression_id = task
    prediction_item = WORKER_RESULTS[video][expression_id]
    video_info = WORKER_META[video]
    expression = video_info["expressions"][expression_id]
    frames = video_info["frames"]
    predictions = prediction_item["prediction_masks"]
    if len(predictions) != len(frames):
        raise ValueError(
            f"{video}/{expression_id}: {len(predictions)} predictions for "
            f"{len(frames)} frames"
        )

    js, fs, gt_present, prediction_present = [], [], [], []
    object_id = str(expression["obj_id"])
    for frame, encoded_prediction in zip(frames, predictions):
        prediction = decode_bool(encoded_prediction)
        if prediction is None:
            raise ValueError(f"{video}/{expression_id}/{frame}: missing prediction")
        annotation_path = WORKER_ANNOTATIONS / video / object_id / f"{frame}.png"
        if annotation_path.exists():
            annotation = np.asarray(Image.open(annotation_path).convert("L")) != 0
            if annotation.shape != prediction.shape:
                raise ValueError(
                    f"{video}/{expression_id}/{frame}: annotation "
                    f"{annotation.shape} != prediction {prediction.shape}"
                )
        else:
            annotation = np.zeros_like(prediction)
        js.append(float(db_eval_iou(annotation, prediction)))
        fs.append(float(db_eval_boundary(annotation, prediction)))
        gt_present.append(bool(annotation.any()))
        prediction_present.append(bool(prediction.any()))

    js_array = np.asarray(js)
    intersection_frames = np.asarray(gt_present) & np.asarray(prediction_present)
    union_frames = np.asarray(gt_present) | np.asarray(prediction_present)
    union = int(union_frames.sum())
    if union == 0:
        temporal_iou = volume_iou = 1.0
    else:
        temporal_iou = float(intersection_frames.sum() / union)
        volume_iou = float(js_array[intersection_frames].sum() / union)
    return {
        "video": video,
        "expression_id": expression_id,
        "type": expression["type"],
        "j": float(js_array.mean()),
        "f": float(np.mean(fs)),
        "tiou": temporal_iou,
        "viou": volume_iou,
    }


def evaluate_frame_chunk(task):
    """Return additive statistics for a bounded frame range.

    Expression-level tasks are badly imbalanced on Long-RVOS: one long 4K clip
    can keep a single worker busy long after every other worker has finished.
    These statistics reproduce ``evaluate_expression`` exactly after merging,
    while allowing chunks from the same expression to use different workers.
    """
    video, expression_id, start, end = task
    prediction_item = WORKER_RESULTS[video][expression_id]
    video_info = WORKER_META[video]
    expression = video_info["expressions"][expression_id]
    frames = video_info["frames"]
    predictions = prediction_item["prediction_masks"]
    if len(predictions) != len(frames):
        raise ValueError(
            f"{video}/{expression_id}: {len(predictions)} predictions for "
            f"{len(frames)} frames"
        )

    sum_j = sum_f = sum_intersection_j = 0.0
    temporal_intersection = temporal_union = 0
    object_id = str(expression["obj_id"])
    for frame, encoded_prediction in zip(
            frames[start:end], predictions[start:end]):
        prediction = decode_bool(encoded_prediction)
        if prediction is None:
            raise ValueError(f"{video}/{expression_id}/{frame}: missing prediction")
        annotation_path = WORKER_ANNOTATIONS / video / object_id / f"{frame}.png"
        if annotation_path.exists():
            annotation = np.asarray(Image.open(annotation_path).convert("L")) != 0
            if annotation.shape != prediction.shape:
                raise ValueError(
                    f"{video}/{expression_id}/{frame}: annotation "
                    f"{annotation.shape} != prediction {prediction.shape}"
                )
        else:
            annotation = np.zeros_like(prediction)
        j = float(db_eval_iou(annotation, prediction))
        sum_j += j
        sum_f += float(db_eval_boundary(annotation, prediction))
        gt_present = bool(annotation.any())
        prediction_present = bool(prediction.any())
        intersection = gt_present and prediction_present
        union = gt_present or prediction_present
        temporal_intersection += int(intersection)
        temporal_union += int(union)
        if intersection:
            sum_intersection_j += j
    return {
        "video": video,
        "expression_id": expression_id,
        "type": expression["type"],
        "frames": end - start,
        "sum_j": sum_j,
        "sum_f": sum_f,
        "temporal_intersection": temporal_intersection,
        "temporal_union": temporal_union,
        "sum_intersection_j": sum_intersection_j,
    }


def merge_frame_chunks(chunks):
    merged = defaultdict(lambda: {
        "frames": 0,
        "sum_j": 0.0,
        "sum_f": 0.0,
        "temporal_intersection": 0,
        "temporal_union": 0,
        "sum_intersection_j": 0.0,
    })
    metadata = {}
    for chunk in chunks:
        key = (chunk["video"], chunk["expression_id"])
        metadata[key] = chunk["type"]
        for field in merged[key]:
            merged[key][field] += chunk[field]

    rows = []
    for (video, expression_id), totals in merged.items():
        count = totals["frames"]
        union = totals["temporal_union"]
        rows.append({
            "video": video,
            "expression_id": expression_id,
            "type": metadata[(video, expression_id)],
            "j": totals["sum_j"] / count,
            "f": totals["sum_f"] / count,
            "tiou": (
                totals["temporal_intersection"] / union if union else 1.0
            ),
            "viou": totals["sum_intersection_j"] / union if union else 1.0,
        })
    return rows


def summarize(rows):
    if not rows:
        return {
            "evaluated_pairs": 0,
            "mean_j": None,
            "mean_f": None,
            "j_and_f": None,
            "tiou": None,
            "viou": None,
        }
    mean_j = float(np.mean([row["j"] for row in rows]))
    mean_f = float(np.mean([row["f"] for row in rows]))
    return {
        "evaluated_pairs": len(rows),
        "mean_j": mean_j,
        "mean_f": mean_f,
        "j_and_f": (mean_j + mean_f) / 2,
        "tiou": float(np.mean([row["tiou"] for row in rows])),
        "viou": float(np.mean([row["viou"] for row in rows])),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--meta", type=Path, default=DEFAULT_ROOT / "meta_expressions.json")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ROOT / "Annotations")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--frame-chunk-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rows-output",
        type=Path,
        help="Optional per-expression metrics for paired diagnostics.",
    )
    return parser.parse_args()


def main():
    global WORKER_META, WORKER_RESULTS, WORKER_ANNOTATIONS
    args = parse_args()
    WORKER_RESULTS = json.loads(args.results.read_text())
    WORKER_META = json.loads(args.meta.read_text())["videos"]
    WORKER_ANNOTATIONS = args.annotations
    if args.frame_chunk_size <= 0:
        raise ValueError("--frame-chunk-size must be positive")
    tasks = [
        (video, expression_id, start, min(start + args.frame_chunk_size, len(WORKER_META[video]["frames"])))
        for video, expressions in WORKER_RESULTS.items()
        if video in WORKER_META
        for expression_id in expressions
        for start in range(
            0, len(WORKER_META[video]["frames"]), args.frame_chunk_size
        )
    ]
    print(f"frame chunks: {len(tasks)}", flush=True)
    chunks = []
    with Pool(processes=args.workers) as pool:
        for index, chunk in enumerate(
                pool.imap_unordered(evaluate_frame_chunk, tasks)):
            chunks.append(chunk)
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(tasks)} frame chunks", flush=True)
    rows = merge_frame_chunks(chunks)
    rows.sort(key=lambda row: (row["video"], row["expression_id"]))

    by_type = {
        kind: summarize([row for row in rows if row["type"] == kind])
        for kind in ("static", "dynamic", "hybrid")
    }
    summary = {
        "protocol": "Official Long-RVOS per-expression J/F, tIoU, and vIoU",
        "results": str(args.results),
        "overall": summarize(rows),
        "by_type": by_type,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

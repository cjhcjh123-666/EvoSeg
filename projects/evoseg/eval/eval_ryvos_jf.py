"""Official Ref-Youtube-VOS J/F evaluation from Sa2VA result JSON.

Metrics use the same DAVIS region-J and boundary-F implementation as the
Long-RVOS evaluator. Scores are averaged over annotated frames per expression
and then over expressions, matching the public RVOS evaluation convention.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_seg.official_metrics import (
    db_eval_boundary,
    db_eval_iou,
)


DEFAULT_ANNOTATIONS = Path(
    "/9950backfile/chenjiahui/evo_artifacts/datasets/"
    "ref_youtube_vos/extracted/valid/Annotations"
)
WORKER_RESULTS = {}
WORKER_ANNOTATIONS = DEFAULT_ANNOTATIONS


def decode_bool(rle):
    return None if rle is None else mask_utils.decode(rle).astype(bool)


def evaluate_expression(task):
    video, expression_id = task
    item = WORKER_RESULTS[video][expression_id]
    predictions = item["prediction_masks"]
    frames = item["frames"]
    if len(predictions) != len(frames):
        raise ValueError(
            f"{video}/{expression_id}: {len(predictions)} predictions for "
            f"{len(frames)} frames"
        )

    js, fs = [], []
    annotation_dir = WORKER_ANNOTATIONS / video / str(expression_id)
    for frame, encoded_prediction in zip(frames, predictions):
        annotation_path = annotation_dir / f"{frame}.png"
        # Ref-Youtube-VOS evaluates the sparsely annotated object frames.
        if not annotation_path.exists():
            continue
        prediction = decode_bool(encoded_prediction)
        if prediction is None:
            raise ValueError(f"{video}/{expression_id}/{frame}: missing prediction")
        annotation = np.asarray(Image.open(annotation_path).convert("L")) != 0
        if prediction.shape != annotation.shape:
            prediction = np.asarray(
                Image.fromarray(prediction.astype(np.uint8)).resize(
                    (annotation.shape[1], annotation.shape[0]),
                    Image.Resampling.NEAREST,
                )
            ).astype(bool)
        js.append(float(db_eval_iou(annotation, prediction)))
        fs.append(float(db_eval_boundary(annotation, prediction)))

    if not js:
        return None
    return {
        "video": video,
        "expression_id": expression_id,
        "j": float(np.mean(js)),
        "f": float(np.mean(fs)),
        "annotated_frames": len(js),
    }


def summarize(rows):
    mean_j = float(np.mean([row["j"] for row in rows]))
    mean_f = float(np.mean([row["f"] for row in rows]))
    return {
        "protocol": "Official Ref-Youtube-VOS per-expression DAVIS J&F",
        "evaluated_pairs": len(rows),
        "evaluated_annotated_frames": int(
            sum(row["annotated_frames"] for row in rows)
        ),
        "mean_j": mean_j,
        "mean_f": mean_f,
        "j_and_f": (mean_j + mean_f) / 2,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    global WORKER_RESULTS, WORKER_ANNOTATIONS
    args = parse_args()
    WORKER_RESULTS = json.loads(args.results.read_text())
    WORKER_ANNOTATIONS = args.annotations
    tasks = [
        (video, expression_id)
        for video, expressions in WORKER_RESULTS.items()
        for expression_id in expressions
    ]
    print(f"tasks: {len(tasks)}", flush=True)
    rows = []
    with Pool(processes=args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(evaluate_expression, tasks)):
            if row is not None:
                rows.append(row)
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(tasks)} expressions", flush=True)
    if not rows:
        raise ValueError("no annotated Ref-Youtube-VOS expressions were evaluated")

    summary = summarize(rows)
    summary["results"] = str(args.results)
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

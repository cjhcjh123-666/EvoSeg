"""MeViS native-resolution J&F with bounded-memory multiprocessing."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_seg.official_metrics import (
    db_eval_boundary,
    db_eval_iou,
)


DEFAULT_META = Path(
    "/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/valid_u/"
    "meta_expressions_v2.json"
)
DEFAULT_MASK = Path(
    "/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/valid_u/"
    "mask_dict.json"
)
WORKER_META = {}
WORKER_MASK = {}
WORKER_RESULTS = {}


def decode_bool(rle):
    return None if rle is None else mask_utils.decode(rle).astype(bool)


def eval_expression(task):
    vid, exp_id = task
    item = WORKER_RESULTS[vid][exp_id]
    exp_info = WORKER_META[vid]["expressions"].get(exp_id)
    if exp_info is None:
        return None
    anno_ids = [str(a) for a in exp_info["anno_id"]]
    js, fs = [], []
    for frame_index, encoded_prediction in enumerate(item["prediction_masks"]):
        prediction = decode_bool(encoded_prediction)
        if prediction is None:
            continue
        target = None
        for anno_id in anno_ids:
            annotations = WORKER_MASK.get(anno_id, [])
            if frame_index >= len(annotations):
                continue
            annotation = decode_bool(annotations[frame_index])
            if annotation is not None:
                target = annotation if target is None else target | annotation
        if target is None:
            target = np.zeros_like(prediction)
        # Official DAVIS-style empty-mask semantics are retained: empty/empty
        # is one, while a false positive on an absent frame is zero.
        js.append(float(db_eval_iou(target, prediction)))
        fs.append(float(db_eval_boundary(target, prediction)))
    if not js:
        return None
    return float(np.mean(js)), float(np.mean(fs)), not anno_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--keys-from",
        type=Path,
        help="Restrict evaluation to video/expression keys present in another results JSON.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    global WORKER_META, WORKER_MASK, WORKER_RESULTS
    args = parse_args()
    WORKER_RESULTS = json.loads(args.results.read_text())
    WORKER_META = json.loads(args.meta.read_text())["videos"]
    WORKER_MASK = json.loads(args.mask.read_text())
    allowed = None
    if args.keys_from:
        key_results = json.loads(args.keys_from.read_text())
        allowed = {
            (video, expression_id)
            for video, expressions in key_results.items()
            for expression_id in expressions
        }
    tasks = [
        (video, exp_id)
        for video, expressions in WORKER_RESULTS.items()
        if video in WORKER_META
        for exp_id in expressions
        if allowed is None or (video, exp_id) in allowed
    ]
    print(f"tasks: {len(tasks)}", flush=True)
    pairs = []
    # Fork workers inherit the read-only dictionaries. Small key-only tasks
    # avoid repeatedly pickling the 190 MB prediction dictionary and prevent
    # long videos with many expressions from creating a worker straggler.
    with Pool(processes=args.workers) as pool:
        for index, score in enumerate(pool.imap_unordered(eval_expression, tasks)):
            if score is not None:
                pairs.append(score)
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(tasks)} expressions", flush=True)
    def summarize(scores):
        if not scores:
            return {"evaluated_pairs": 0, "mean_j": None, "mean_f": None,
                    "j_and_f": None}
        js = [score[0] for score in scores]
        fs = [score[1] for score in scores]
        return {
            "evaluated_pairs": len(scores),
            "mean_j": float(np.mean(js)),
            "mean_f": float(np.mean(fs)),
            "j_and_f": float((np.mean(js) + np.mean(fs)) / 2),
        }

    overall = summarize(pairs)
    target_present = summarize([score for score in pairs if not score[2]])
    no_target = summarize([score for score in pairs if score[2]])
    summary = {
        "protocol": "DAVIS J&F (Long-RVOS vendored official implementation)",
        "empty_frame_semantics": "empty/empty=1; false-positive-on-empty=0",
        "results": str(args.results),
        "keys_from": str(args.keys_from) if args.keys_from else None,
        **overall,
        "target_present": target_present,
        "no_target": no_target,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

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


def decode_bool(rle):
    return None if rle is None else mask_utils.decode(rle).astype(bool)


def eval_video(task):
    vid, exps = task
    meta = WORKER_META[vid]["expressions"]
    out = []
    for exp_id, item in exps.items():
        exp_info = meta.get(exp_id)
        if exp_info is None:
            continue
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
        if js:
            out.append((float(np.mean(js)), float(np.mean(fs))))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    global WORKER_META, WORKER_MASK
    args = parse_args()
    results = json.loads(args.results.read_text())
    WORKER_META = json.loads(args.meta.read_text())["videos"]
    WORKER_MASK = json.loads(args.mask.read_text())
    tasks = [(video, expressions) for video, expressions in results.items()
             if video in WORKER_META]
    print(f"tasks: {len(tasks)}", flush=True)
    pairs = []
    # Fork workers inherit the read-only dictionaries. Tasks carry only one
    # video's predictions, not the full 400+ MB mask dictionary.
    with Pool(processes=args.workers) as pool:
        for index, scores in enumerate(pool.imap_unordered(eval_video, tasks)):
            pairs.extend(scores)
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(tasks)} videos", flush=True)
    js = [score[0] for score in pairs]
    fs = [score[1] for score in pairs]
    summary = {
        "protocol": "DAVIS J&F (Long-RVOS vendored official implementation)",
        "empty_frame_semantics": "empty/empty=1; false-positive-on-empty=0",
        "results": str(args.results),
        "evaluated_pairs": len(pairs),
        "mean_j": float(np.mean(js)),
        "mean_f": float(np.mean(fs)),
        "j_and_f": float((np.mean(js) + np.mean(fs)) / 2),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

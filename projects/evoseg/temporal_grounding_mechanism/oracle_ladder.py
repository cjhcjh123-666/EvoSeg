"""Build the GT-separated Oracle Ladder on the fixed pilot-64 expressions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_grounding_interface.protocol import (
    deterministic_positive_point,
    mask_iou,
)
from projects.evoseg.temporal_grounding_mechanism.common import identity, write_csv
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


STAGES = (1, 3, 5, 7)


def decode(value: dict) -> np.ndarray:
    rle = dict(value)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def load_gt(item: dict, shape: tuple[int, int]) -> np.ndarray:
    values = []
    for path_value, present in zip(item["evaluation_mask_paths"], item["evaluation_mask_present"]):
        path = Path(path_value)
        if bool(present) != path.is_file():
            raise RuntimeError(f"GT availability changed: {path}")
        if present:
            with Image.open(path) as image:
                value = np.asarray(image.convert("L")) > 0
        else:
            value = np.zeros(shape, dtype=bool)
        values.append(value)
    return np.stack(values)


def metrics(gt: np.ndarray, prediction: np.ndarray) -> dict:
    j = float(np.mean(db_eval_iou(gt, prediction)))
    f = float(np.mean(db_eval_boundary(gt, prediction)))
    return {"J": j, "F": f, "J_and_F": (j + f) / 2}


def candidate_sources(root: Path) -> dict[str, Path]:
    result = {}
    for shard in sorted(root.glob("shard-*")):
        path = shard / "candidate_records.jsonl"
        if not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "success" or row.get("prompt_method") != "concept":
                    continue
                key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                result[key] = shard / row["candidate_tracks_path"]
    return result


def state_sources(path: Path) -> dict[str, Path]:
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "success":
                result[row["identity"]] = Path(row["state_path"])
    return result


def choose_tracks(
    item: dict, states, tracks: dict, state_kind: str
) -> tuple[list[int | None], list[dict], np.ndarray]:
    evaluation = tracks["evaluation_frame_indices"]
    decoded = [[decode(value) for value in track["frames"]] for track in tracks["tracks"]]
    chosen = []
    plan = []
    for stage in STAGES:
        endpoint = int(round((stage + 1) * item["frame_count"] / 8) - 1)
        position = min(range(len(evaluation)), key=lambda index: abs(evaluation[index] - endpoint))
        anchor = int(evaluation[position])
        reference = np.asarray(states[f"{state_kind}_{stage}_mask"], dtype=bool)
        candidates = [sequence[position] for sequence in decoded]
        scores = [mask_iou(reference, value) for value in candidates]
        selected = int(np.argmax(scores)) if scores else None
        track_id = int(tracks["tracks"][selected]["track_id"]) if selected is not None else None
        mask = candidates[selected] if selected is not None else np.zeros(reference.shape, dtype=bool)
        point = deterministic_positive_point(mask) if mask.any() else None
        chosen.append(track_id)
        plan.append(
            {
                "stage_order": len(plan),
                "canonical_stage_index": stage,
                "anchor_frame_index": anchor,
                "state_kind": state_kind,
                "candidate_count": len(candidates),
                "selected_candidate_index": selected,
                "selected_candidate_object_id": track_id,
                "selected_candidate_confidence": tracks["tracks"][selected].get("mean_confidence") if selected is not None else None,
                "candidate_scores": scores,
                "update_applied": point is not None,
                "positive_point_relative_xy": point,
                "nearest_saved_track_frame_to_nominal_stage_endpoint": anchor,
                "nominal_stage_endpoint": endpoint,
            }
        )
    direct = []
    for frame in evaluation:
        stage_position = next(
            (index for index, row in enumerate(plan) if frame <= row["anchor_frame_index"]),
            len(plan) - 1,
        )
        track_id = chosen[stage_position]
        track_position = next(
            (index for index, track in enumerate(tracks["tracks"]) if int(track["track_id"]) == track_id),
            None,
        )
        evaluation_position = evaluation.index(frame)
        direct.append(
            decoded[track_position][evaluation_position]
            if track_position is not None
            else np.zeros((tracks["height"], tracks["width"]), dtype=bool)
        )
    return chosen, plan, np.stack(direct)


def run(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    items = manifest["objects"][: args.max_objects]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid object shard")
    items = [
        item for index, item in enumerate(items) if index % args.num_shards == args.shard_index
    ]
    candidates = candidate_sources(Path(args.candidate_root))
    states = state_sources(Path(args.state_records))
    metric_rows = {}
    with Path(args.candidate_metrics).open() as handle:
        for row in csv.DictReader(handle):
            if row["prompt_method"] != "concept":
                continue
            key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
            metric_rows[key] = row
    output = []
    plans = []
    for item in items:
        for expression in item["expressions"]:
            key = identity(item["dataset"], item["video_id"], item["object_id"], expression["expression_id"])
            if key not in candidates or key not in states or key not in metric_rows:
                raise RuntimeError(f"missing cached input for {key}")
            tracks = json.loads(candidates[key].read_text())
            shape = (int(tracks["height"]), int(tracks["width"]))
            gt = load_gt(item, shape)
            candidate_metric = metric_rows[key]
            values = json.loads(candidate_metric["candidate_metrics"])
            oracle_id = int(candidate_metric["oracle_track_id"]) if candidate_metric["oracle_track_id"] else None
            oracle_value = next((value for value in values if int(value["track_id"]) == oracle_id), None)
            base = {
                "identity": key,
                "dataset": item["dataset"],
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "description_type": expression["type"],
                "expression": expression["text"],
                "prompt_method": "concept",
                "candidate_count": len(tracks["tracks"]),
            }
            output.append(
                {
                    **base,
                    "condition": "ORACLE_CANDIDATE",
                    "selected_track_ids": json.dumps([oracle_id]),
                    "J": float(oracle_value["J"]) if oracle_value else 0.0,
                    "F": float(oracle_value["F"]) if oracle_value else 0.0,
                    "J_and_F": float(oracle_value["J_and_F"]) if oracle_value else 0.0,
                    "ORACLE": True,
                }
            )
            with np.load(states[key]) as state_values:
                for state_kind in ("static", "temporal"):
                    selected, plan, prediction = choose_tracks(item, state_values, tracks, state_kind)
                    score = metrics(gt, prediction)
                    condition = f"PREDICTED_CANDIDATE_DIRECT_{state_kind.upper()}"
                    output.append(
                        {
                            **base,
                            "condition": condition,
                            "selected_track_ids": json.dumps(selected),
                            **score,
                            "ORACLE": False,
                        }
                    )
                    plans.append({**base, "condition": f"PREDICTED_CANDIDATE_POINT_{state_kind.upper()}", "ORACLE": False, "plan": plan})
            oracle_plan = []
            oracle_track = next((track for track in tracks["tracks"] if int(track["track_id"]) == oracle_id), None)
            for order, stage in enumerate(STAGES):
                endpoint = int(round((stage + 1) * item["frame_count"] / 8) - 1)
                position = min(range(len(tracks["evaluation_frame_indices"])), key=lambda index: abs(tracks["evaluation_frame_indices"][index] - endpoint))
                anchor = int(tracks["evaluation_frame_indices"][position])
                mask = decode(oracle_track["frames"][position]) if oracle_track else np.zeros(shape, dtype=bool)
                point = deterministic_positive_point(mask) if mask.any() else None
                oracle_plan.append(
                    {
                        "stage_order": order,
                        "canonical_stage_index": stage,
                        "anchor_frame_index": anchor,
                        "state_kind": "ORACLE_ID",
                        "candidate_count": len(tracks["tracks"]),
                        "selected_candidate_index": None,
                        "selected_candidate_object_id": oracle_id,
                        "selected_candidate_confidence": oracle_track.get("mean_confidence") if oracle_track else None,
                        "candidate_scores": [],
                        "update_applied": point is not None,
                        "positive_point_relative_xy": point,
                        "nearest_saved_track_frame_to_nominal_stage_endpoint": anchor,
                        "nominal_stage_endpoint": endpoint,
                    }
                )
            plans.append({**base, "condition": "ORACLE_ID_POINT", "ORACLE": True, "plan": oracle_plan})
    write_csv(Path(args.output_csv), output)
    plan_path = Path(args.output_plans)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w") as handle:
        for row in plans:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"expressions": len(output) // 3, "rows": len(output), "plans": len(plans)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--state-records", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plans", required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

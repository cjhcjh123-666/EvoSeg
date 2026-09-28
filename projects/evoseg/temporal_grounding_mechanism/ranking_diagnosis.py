"""Diagnose whether Sa2VA N=8/16/32 changes SAM3.1 candidate ranking."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.stats import spearmanr

from projects.evoseg.temporal_grounding_interface.protocol import mask_iou
from projects.evoseg.temporal_grounding_mechanism.common import identity, write_csv


def decode(value: dict) -> np.ndarray:
    rle = dict(value)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def prediction_index(root: Path) -> dict[tuple[str, int], dict]:
    result = {}
    with (root / "predictions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
            result[(key, int(row["frame_budget"]))] = row
    return result


def candidate_index(root: Path) -> dict[str, Path]:
    result = {}
    for shard in sorted(path for path in root.glob("shard-*") if path.is_dir()):
        with (shard / "candidate_records.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success" and row.get("prompt_method") == "concept":
                    key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                    result[key] = shard / row["candidate_tracks_path"]
    return result


def representation_index(root: Path) -> dict[tuple[str, int], Path]:
    result = {}
    with (root / "representation_records.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "success":
                key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                result[(key, int(row["frame_budget"]))] = root / row["vector_path"]
    return result


def bootstrap_spearman(rows: list[dict], x: str, y: str, iterations: int = 2000) -> dict:
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    videos = sorted(by_video)
    observed = float(spearmanr([r[x] for r in rows], [r[y] for r in rows]).statistic)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(videos, len(videos), replace=True)
        sample = [row for video in sampled for row in by_video[video]]
        value = float(spearmanr([r[x] for r in sample], [r[y] for r in sample]).statistic)
        if np.isfinite(value):
            values.append(value)
    return {
        "x": x,
        "y": y,
        "rho": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "expressions": len(rows),
        "videos": len(videos),
        "iterations": iterations,
    }


def run(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    allowed = {
        identity(item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]): (item, expression)
        for item in manifest["objects"][: args.max_objects]
        for expression in item["expressions"]
        if expression["type"] == "dynamic"
    }
    predictions = prediction_index(Path(args.segmentation_root))
    candidates = candidate_index(Path(args.candidate_root))
    representations = representation_index(Path(args.representation_root))
    oracle = {}
    with Path(args.candidate_metrics).open() as handle:
        for row in csv.DictReader(handle):
            if row["prompt_method"] == "concept":
                key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                oracle[key] = int(row["oracle_track_id"]) if row["oracle_track_id"] else None
    output = []
    delta_rows = []
    for key, (item, expression) in sorted(allowed.items()):
        track_value = json.loads(candidates[key].read_text())
        track_ids = [int(track["track_id"]) for track in track_value["tracks"]]
        correct = oracle[key]
        budget_rows = {}
        for budget in (8, 16, 32):
            prediction_row = predictions[(key, budget)]
            mask_rows = json.loads((Path(args.segmentation_root) / prediction_row["rle_masks_path"]).read_text())
            predicted = [decode(row["rle"]) for row in mask_rows]
            scores = []
            for track in track_value["tracks"]:
                candidate_masks = [decode(value) for value in track["frames"]]
                scores.append(float(np.mean([mask_iou(left, right) for left, right in zip(predicted, candidate_masks)])))
            order = np.argsort(-np.asarray(scores)).tolist() if scores else []
            correct_position = track_ids.index(correct) if correct in track_ids else None
            correct_rank = order.index(correct_position) + 1 if correct_position is not None else None
            correct_score = scores[correct_position] if correct_position is not None else 0.0
            distractors = [score for index, score in enumerate(scores) if index != correct_position]
            margin = correct_score - max(distractors, default=0.0)
            row = {
                "identity": key,
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "description_type": "dynamic",
                "expression": expression["text"],
                "frame_budget": budget,
                "oracle_correct_track_id": correct,
                "correct_candidate_rank": correct_rank,
                "correct_candidate_score": correct_score,
                "top1_candidate_track_id": track_ids[order[0]] if order else None,
                "top1_is_correct": int(bool(order) and track_ids[order[0]] == correct),
                "target_vs_best_distractor_margin": margin,
                "segmentation_J_and_F": float(prediction_row["J_and_F"]),
                "candidate_count": len(track_ids),
            }
            output.append(row)
            budget_rows[budget] = row
        with np.load(representations[(key, 8)]) as left, np.load(representations[(key, 32)]) as right:
            z8 = np.asarray(left["z_seg"][0], dtype=np.float64)
            z32 = np.asarray(right["z_seg"][0], dtype=np.float64)
        cosine = float(np.dot(z8, z32) / (np.linalg.norm(z8) * np.linalg.norm(z32)))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        delta_rows.append(
            {
                "identity": key,
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "representation_cosine_distance_8_32": 1.0 - cosine,
                "representation_angular_distance_8_32": float(np.arccos(cosine) / np.pi),
                "representation_normalized_l2_8_32": float(np.linalg.norm(z32 / np.linalg.norm(z32) - z8 / np.linalg.norm(z8))),
                "candidate_margin_change_32_minus_8": budget_rows[32]["target_vs_best_distractor_margin"] - budget_rows[8]["target_vs_best_distractor_margin"],
                "correct_rank_change_32_minus_8": (budget_rows[32]["correct_candidate_rank"] or len(track_ids) + 1) - (budget_rows[8]["correct_candidate_rank"] or len(track_ids) + 1),
                "top1_identity_changed_8_32": int(budget_rows[8]["top1_candidate_track_id"] != budget_rows[32]["top1_candidate_track_id"]),
                "J_and_F_change_32_minus_8": budget_rows[32]["segmentation_J_and_F"] - budget_rows[8]["segmentation_J_and_F"],
            }
        )
    write_csv(Path(args.output), output)
    write_csv(Path(args.delta_output), delta_rows)
    correlations = [
        bootstrap_spearman(delta_rows, "representation_cosine_distance_8_32", "candidate_margin_change_32_minus_8"),
        bootstrap_spearman(delta_rows, "representation_cosine_distance_8_32", "J_and_F_change_32_minus_8"),
        bootstrap_spearman(delta_rows, "candidate_margin_change_32_minus_8", "J_and_F_change_32_minus_8"),
    ]
    Path(args.correlations).write_text(json.dumps(correlations, indent=2) + "\n")
    print(json.dumps({"budget_rows": len(output), "dynamic_expressions": len(delta_rows), "correlations": correlations}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--segmentation-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--representation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-output", required=True)
    parser.add_argument("--correlations", required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

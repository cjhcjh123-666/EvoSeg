#!/usr/bin/env python3
"""Render prediction-independent real cases for the minimal matcher prototype."""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from projects.evoseg.temporal_compiler.temporal_matcher_prototype import (
    decode_rle,
    load_successful_candidate_records,
)


def overlay(image: np.ndarray, ground_truth: np.ndarray, prediction=None) -> np.ndarray:
    value = image.astype(np.float32).copy()
    value[ground_truth] = 0.55 * value[ground_truth] + 0.45 * np.array([20, 210, 80])
    if prediction is not None:
        value[prediction] = 0.55 * value[prediction] + 0.45 * np.array([235, 45, 80])
    return np.clip(value, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    objects = {
        (item["dataset"], str(item["video_id"]), str(item["object_id"])): item
        for item in manifest["objects"]
    }
    candidates = load_successful_candidate_records(args.candidate_root, "concept")
    predictions = []
    seeds = []
    for path in args.prediction:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        file_seeds = {int(row["seed"]) for row in rows}
        if len(file_seeds) != 1:
            raise RuntimeError(f"prediction file is not one seed: {path}")
        predictions.append({row["identity"]: row for row in rows})
        seeds.append(next(iter(file_seeds)))
    identities = sorted(
        identity
        for identity, row in predictions[0].items()
        if row["description_type"] == "dynamic"
        and all(identity in values for values in predictions)
    )
    random.Random(args.selection_seed).shuffle(identities)
    selected = sorted(identities[: args.cases])
    if len(selected) != args.cases:
        raise RuntimeError("not enough complete Dynamic test expressions")

    columns = ["RGB", "GT"] + [label for seed in seeds for label in (f"C s{seed}", f"D s{seed}")]
    fig, axes = plt.subplots(
        len(selected), len(columns), figsize=(2.25 * len(columns), 2.8 * len(selected)), squeeze=False
    )
    audit_cases = []
    for row_index, identity_string in enumerate(selected):
        identity = tuple(identity_string.split("/"))
        candidate = candidates[identity]
        tracks = json.loads(Path(candidate["candidate_tracks_absolute_path"]).read_text())
        item = objects[identity[:3]]
        if tracks["evaluation_frame_indices"] != item["evaluation_frame_indices"]:
            raise AssertionError("candidate and manifest evaluation frames differ")
        midpoint = len(tracks["evaluation_frame_indices"]) // 2
        present_positions = [
            index
            for index, present in enumerate(item["evaluation_mask_present"])
            if present
        ]
        if not present_positions:
            raise RuntimeError(f"selected case has no evaluation GT: {identity_string}")
        position = min(present_positions, key=lambda index: abs(index - midpoint))
        frame_index = tracks["evaluation_frame_indices"][position]
        frame_name = item["frame_names"][frame_index]
        image_path = Path(manifest["dataset"]["image_root"]) / identity[1] / f"{frame_name}.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        ground_truth = np.asarray(
            Image.open(item["evaluation_mask_paths"][position]).convert("L")
        ) > 0
        track_masks = {
            int(track["track_id"]): decode_rle(track["frames"][position])
            for track in tracks["tracks"]
        }
        panels = [image, overlay(image, ground_truth)]
        selected_tracks = {}
        for seed, values in zip(seeds, predictions):
            row = values[identity_string]
            for method in ("static", "temporal"):
                track_value = row[f"{method}_track_id"]
                track_id = int(track_value) if track_value != "" else None
                selected_tracks[f"{method}_seed{seed}"] = track_id
                prediction = (
                    track_masks[track_id]
                    if track_id is not None
                    else np.zeros_like(ground_truth)
                )
                panels.append(overlay(image, ground_truth, prediction))
        for column_index, (title, panel) in enumerate(zip(columns, panels)):
            axis = axes[row_index, column_index]
            axis.imshow(panel)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        expression = predictions[0][identity_string]["expression"]
        axes[row_index, 0].set_ylabel(
            "\n".join(textwrap.wrap(expression, 34)), fontsize=8
        )
        audit_cases.append(
            {
                "identity": identity_string,
                "expression": expression,
                "concept_prompt": candidate["prompt"],
                "frame_index": frame_index,
                "frame_name": frame_name,
                "selected_tracks": selected_tracks,
            }
        )
    fig.suptitle(
        "Fixed-seed Dynamic test cases (green=GT, red=selected candidate)", fontsize=12
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selection": "seeded shuffle of all Dynamic test identities; prediction-independent",
                "selection_seed": args.selection_seed,
                "model_seeds": seeds,
                "cases": audit_cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Matched, object-balanced VIRST/ProcessVIRST comparison on one manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.process_virst.long_rvos_train_adapter import ORDER_PATTERN


def load_predictions(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["key"]
                if key in rows:
                    raise RuntimeError(f"duplicate prediction key: {key}")
                rows[key] = row
    return rows


def expected_rows(manifest: dict) -> list[dict]:
    rows = []
    for item in manifest["objects"]:
        for expression in item["expressions"]:
            rows.append(
                {
                    "key": "/".join(
                        [
                            item["dataset"],
                            item["video_id"],
                            str(item["object_id"]),
                            str(expression["expression_id"]),
                            "native",
                        ]
                    ),
                    "dataset": item["dataset"],
                    "video_id": item["video_id"],
                    "object_id": str(item["object_id"]),
                    "expression_id": str(expression["expression_id"]),
                    "description_type": expression["type"].lower(),
                    "expression": expression["text"],
                    "explicit_order": bool(ORDER_PATTERN.search(expression["text"])),
                }
            )
    return rows


def bootstrap_object_delta(rows: list[dict], iterations: int, seed: int) -> dict:
    by_object = defaultdict(list)
    for row in rows:
        by_object[(row["video_id"], row["object_id"])].append(row["delta_J_and_F"])
    object_rows = [
        {"video_id": key[0], "delta": float(np.mean(values))}
        for key, values in by_object.items()
    ]
    by_video = defaultdict(list)
    for row in object_rows:
        by_video[row["video_id"]].append(row["delta"])
    videos = sorted(by_video)
    point = float(np.mean([row["delta"] for row in object_rows]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        chosen = rng.choice(videos, size=len(videos), replace=True)
        samples.append(
            np.mean([value for video in chosen for value in by_video[str(video)]])
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta_J_and_F": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "objects": len(object_rows),
        "source_videos": len(videos),
        "bootstrap_unit": "source_video",
        "bootstrap_iterations": iterations,
    }


def object_balanced(rows: list[dict], prefix: str) -> dict:
    by_object = defaultdict(list)
    for row in rows:
        by_object[(row["video_id"], row["object_id"])].append(row)
    result = {}
    for metric in ("J", "F", "J_and_F"):
        result[metric] = float(
            np.mean(
                [
                    np.mean([row[f"{prefix}_{metric}"] for row in object_rows])
                    for object_rows in by_object.values()
                ]
            )
        )
    return result


def compare(
    manifest_path: Path,
    baseline_paths: list[Path],
    process_paths: list[Path],
    output_csv: Path,
    output_json: Path,
    iterations: int,
    seed: int,
) -> dict:
    expected = expected_rows(json.loads(manifest_path.read_text()))
    baseline = load_predictions(baseline_paths)
    process = load_predictions(process_paths)
    rows = []
    missing = []
    for item in expected:
        left = baseline.get(item["key"])
        right = process.get(item["key"])
        if left is None or right is None or left.get("status") != "success" or right.get("status") != "success":
            missing.append(item["key"])
            continue
        row = dict(item)
        for name, value in (("baseline", left), ("process", right)):
            for metric in ("J", "F", "J_and_F"):
                row[f"{name}_{metric}"] = float(value[metric])
        row["delta_J_and_F"] = row["process_J_and_F"] - row["baseline_J_and_F"]
        rows.append(row)
    if missing:
        raise RuntimeError(f"incomplete matched comparison, missing {len(missing)}: {missing[:5]}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {
        "static": [row for row in rows if row["description_type"] == "static"],
        "dynamic": [row for row in rows if row["description_type"] == "dynamic"],
        "hybrid": [row for row in rows if row["description_type"] == "hybrid"],
        "explicit_order": [row for row in rows if row["explicit_order"]],
    }
    summary = {"matched_expressions": len(rows), "missing_or_failed": 0, "groups": {}}
    for name, values in groups.items():
        if not values:
            summary["groups"][name] = {"status": "unavailable_no_samples"}
            continue
        summary["groups"][name] = {
            "status": "success",
            "expressions": len(values),
            "baseline": object_balanced(values, "baseline"),
            "process": object_balanced(values, "process"),
            **bootstrap_object_delta(values, iterations, seed),
        }
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--process", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.manifest,
                args.baseline,
                args.process,
                args.output_csv,
                args.output_json,
                args.bootstrap_iterations,
                args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()

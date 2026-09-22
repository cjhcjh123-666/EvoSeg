"""Summarize within-model Static/Dynamic gaps on one fixed manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SUCCESS = {"success", "success_no_seg"}


def read_jsonl(path: Path) -> list[dict]:
    by_key = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_key[record["key"]] = record
    return list(by_key.values())


def expression_identity(record: dict) -> tuple[str, str, str, str]:
    return (
        record["dataset"],
        record["video_id"],
        str(record["object_id"]),
        str(record["expression_id"]),
    )


def manifest_identities(manifest: dict) -> set[tuple[str, str, str, str]]:
    return {
        (
            item["dataset"],
            item["video_id"],
            str(item["object_id"]),
            str(expression["expression_id"]),
        )
        for item in manifest["objects"]
        for expression in item["expressions"]
    }


def cluster_bootstrap(rows: list[dict], iterations: int, seed: int) -> tuple[float, float, float]:
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row["dynamic_minus_static_J_and_F"])
    videos = sorted(by_video)
    point = float(np.mean([value for values in by_video.values() for value in values]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(iterations):
        selected = rng.choice(videos, size=len(videos), replace=True)
        boot.append(
            float(np.mean([value for video in selected for value in by_video[str(video)]]))
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def summarize_model(spec: dict, expected: set, iterations: int, seed: int):
    records = read_jsonl(Path(spec["predictions"]))
    if spec.get("frame_budget") is not None:
        records = [
            record
            for record in records
            if int(record.get("frame_budget", spec["frame_budget"]))
            == int(spec["frame_budget"])
        ]
    successful = [record for record in records if record.get("status") in SUCCESS]
    succeeded_identities = {expression_identity(record) for record in successful}
    failed = [record for record in records if record.get("status") not in SUCCESS]
    grouped = defaultdict(list)
    for record in successful:
        grouped[
            (
                record["video_id"],
                str(record["object_id"]),
                record["description_type"],
            )
        ].append(record)
    object_means = {
        key: {
            metric: float(np.mean([record[metric] for record in values]))
            for metric in ("J", "F", "J_and_F")
        }
        for key, values in grouped.items()
    }
    pair_rows = []
    objects = sorted({(video, obj) for video, obj, _ in object_means})
    for video_id, object_id in objects:
        static = object_means.get((video_id, object_id, "static"))
        dynamic = object_means.get((video_id, object_id, "dynamic"))
        if static is None or dynamic is None:
            continue
        pair_rows.append(
            {
                "model": spec["name"],
                "video_id": video_id,
                "object_id": object_id,
                **{f"static_{metric}": static[metric] for metric in static},
                **{f"dynamic_{metric}": dynamic[metric] for metric in dynamic},
                "dynamic_minus_static_J_and_F": dynamic["J_and_F"]
                - static["J_and_F"],
                "dynamic_worse": int(dynamic["J_and_F"] < static["J_and_F"]),
            }
        )
    if not pair_rows:
        raise RuntimeError(f"no complete Static/Dynamic object pairs for {spec['name']}")
    point, low, high = cluster_bootstrap(pair_rows, iterations, seed)
    summary = {
        "model": spec["name"],
        "status": "success",
        "paired_objects": len(pair_rows),
        "source_videos": len({row["video_id"] for row in pair_rows}),
        "static_J": float(np.mean([row["static_J"] for row in pair_rows])),
        "static_F": float(np.mean([row["static_F"] for row in pair_rows])),
        "static_J_and_F": float(
            np.mean([row["static_J_and_F"] for row in pair_rows])
        ),
        "dynamic_J": float(np.mean([row["dynamic_J"] for row in pair_rows])),
        "dynamic_F": float(np.mean([row["dynamic_F"] for row in pair_rows])),
        "dynamic_J_and_F": float(
            np.mean([row["dynamic_J_and_F"] for row in pair_rows])
        ),
        "dynamic_minus_static_J_and_F": point,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "dynamic_worse_object_fraction": float(
            np.mean([row["dynamic_worse"] for row in pair_rows])
        ),
        "bootstrap_unit": "source_video",
        "bootstrap_iterations": iterations,
        "input_frames": spec.get("input_frames"),
        "resolution": spec.get("resolution"),
        "checkpoint": spec.get("checkpoint"),
        "checkpoint_source": spec.get("checkpoint_source"),
        "training_data": spec.get("training_data"),
        "explicit_temporal_module": spec.get("explicit_temporal_module"),
    }
    audit = {
        "model": spec["name"],
        "records_after_condition_filter": len(records),
        "successful_records": len(successful),
        "failed_records": len(failed),
        "expected_expression_identities": len(expected),
        "successful_expression_identities": len(succeeded_identities),
        "missing_expression_identities": [
            "/".join(value) for value in sorted(expected - succeeded_identities)
        ],
        "extra_expression_identities": [
            "/".join(value) for value in sorted(succeeded_identities - expected)
        ],
        "failed_keys": [record["key"] for record in failed],
    }
    return summary, pair_rows, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args) -> None:
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())
    expected = manifest_identities(manifest)
    summaries = []
    pairs = []
    audits = []
    for spec in config["models"]:
        if spec.get("status") == "blocked":
            summaries.append(
                {
                    "model": spec["name"],
                    "status": "blocked",
                    "blocker": spec["blocker"],
                    "checkpoint": spec.get("checkpoint"),
                    "checkpoint_source": spec.get("checkpoint_source"),
                    "explicit_temporal_module": spec.get("explicit_temporal_module"),
                }
            )
            audits.append({"model": spec["name"], "status": "blocked", "blocker": spec["blocker"]})
            continue
        summary, model_pairs, audit = summarize_model(
            spec, expected, args.bootstrap_iterations, args.seed
        )
        summaries.append(summary)
        pairs.extend(model_pairs)
        audits.append(audit)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "cross_model_summary.csv", summaries)
    write_csv(output_dir / "cross_model_object_pairs.csv", pairs)
    atomic = output_dir / "cross_model_audit.json"
    atomic.write_text(json.dumps(audits, indent=2, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

"""Summarize the preregistered GroundMoRe ProcessVIRST pilot gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_spec(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=CSV_PATH")
    return label, Path(path)


def cluster_bootstrap(
    differences: list[tuple[str, float]], iterations: int, seed: int
) -> tuple[float, float]:
    by_video: dict[str, list[float]] = defaultdict(list)
    for video, difference in differences:
        by_video[video].append(difference)
    videos = sorted(by_video)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = generator.choice(videos, size=len(videos), replace=True)
        values = [value for video in sampled for value in by_video[str(video)]]
        estimates.append(float(np.mean(values)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_spec, required=True)
    parser.add_argument("--order", action="append", type=parse_spec, default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_rows = read_rows(args.baseline)
    baseline = {
        (row["video_id"], row["expression_id"]): row
        for row in baseline_rows
        if row["status"] == "success"
    }
    summaries = []
    for label, path in args.condition:
        rows = [row for row in read_rows(path) if row["status"] == "success"]
        matched = []
        for row in rows:
            key = (row["video_id"], row["expression_id"])
            if key in baseline:
                matched.append(
                    (
                        row["video_id"],
                        float(row["J_and_F"]) - float(baseline[key]["J_and_F"]),
                    )
                )
        if len(matched) != len(baseline):
            raise RuntimeError(
                f"{label}: matched {len(matched)}/{len(baseline)} baseline expressions"
            )
        low, high = cluster_bootstrap(
            matched, args.bootstrap_iterations, args.seed
        )
        summaries.append(
            {
                "condition": label,
                "expressions": len(rows),
                "source_videos": len({row["video_id"] for row in rows}),
                "J": float(np.mean([float(row["J"]) for row in rows])),
                "F": float(np.mean([float(row["F"]) for row in rows])),
                "J_and_F": float(
                    np.mean([float(row["J_and_F"]) for row in rows])
                ),
                "delta_J_and_F_vs_official": float(
                    np.mean([value for _, value in matched])
                ),
                "delta_ci_low": low,
                "delta_ci_high": high,
            }
        )

    order_summary = {}
    for label, path in args.order:
        rows = read_rows(path)
        by_permutation: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_permutation[row["permutation"]].append(row)
        order_summary[label] = {
            permutation: {
                "expressions": len(values),
                "margin": float(
                    np.mean([float(row["order_preference_margin"]) for row in values])
                ),
                "pre_prompt_response_l2": float(
                    np.mean([float(row["pre_prompt_response_l2"]) for row in values])
                ),
            }
            for permutation, values in sorted(by_permutation.items())
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "baseline_expressions": len(baseline),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_unit": "source_video",
        "conditions": summaries,
        "order": order_summary,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

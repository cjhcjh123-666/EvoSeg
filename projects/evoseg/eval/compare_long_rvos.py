"""Paired Long-RVOS comparison with source-video cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    return {(row["video"], str(row["expression_id"])): row for row in rows}


def _metric(row, name):
    if name == "j_and_f":
        return (row["j"] + row["f"]) / 2
    return row[name]


def paired_deltas(baseline, candidate, expression_type=None, metric="j_and_f"):
    keys = sorted(set(baseline) & set(candidate))
    values = []
    for key in keys:
        left, right = baseline[key], candidate[key]
        if left["type"] != right["type"]:
            raise ValueError(f"type mismatch for {key}")
        if expression_type is not None and left["type"] != expression_type:
            continue
        values.append({
            "video": key[0],
            "expression_id": key[1],
            "delta": _metric(right, metric) - _metric(left, metric),
        })
    return values


def cluster_bootstrap_mean(values, samples=10000, seed=42):
    if not values:
        return [None, None]
    by_video = defaultdict(list)
    for row in values:
        by_video[row["video"]].append(row["delta"])
    videos = sorted(by_video)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.choice(videos, size=len(videos), replace=True)
        sample = [delta for video in chosen for delta in by_video[video]]
        estimates[index] = np.mean(sample)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def summarize(values, bootstrap_samples=10000, seed=42):
    deltas = np.asarray([row["delta"] for row in values], dtype=np.float64)
    if not len(deltas):
        return {"n": 0}
    low, high = cluster_bootstrap_mean(values, bootstrap_samples, seed)
    return {
        "n": int(len(deltas)),
        "videos": len({row["video"] for row in values}),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "positive_fraction": float((deltas > 0).mean()),
        "large_win_fraction": float((deltas > 0.1).mean()),
        "large_loss_fraction": float((deltas < -0.1).mean()),
        "video_cluster_bootstrap_95_ci": [float(low), float(high)],
    }


def compare(baseline, candidate, bootstrap_samples=10000, seed=42):
    output = {}
    for metric in ("j_and_f", "j", "f", "tiou", "viou"):
        output[metric] = {}
        for expression_type in (None, "static", "dynamic", "hybrid"):
            label = "overall" if expression_type is None else expression_type
            values = paired_deltas(
                baseline, candidate, expression_type, metric=metric
            )
            output[metric][label] = summarize(
                values, bootstrap_samples=bootstrap_samples, seed=seed
            )
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    result = {
        "protocol": "paired expression deltas with source-video cluster bootstrap",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "metrics": compare(
            load_rows(args.baseline),
            load_rows(args.candidate),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()

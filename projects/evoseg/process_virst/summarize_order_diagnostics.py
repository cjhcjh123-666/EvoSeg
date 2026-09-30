"""Summarize original/permuted ProcessVIRST alignment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scalar(value) -> float:
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected a singleton diagnostic value, got {value}")
        value = value[0]
    return float(value)


def bootstrap(rows: list[dict], key: str, iterations: int, seed: int) -> dict:
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(float(row[key]))
    videos = sorted(by_video)
    values = [value for group in by_video.values() for value in group]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        chosen = rng.choice(videos, size=len(videos), replace=True)
        samples.append(np.mean([value for video in chosen for value in by_video[str(video)]]))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "expressions": len(rows),
        "source_videos": len(videos),
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "source_video",
    }


def summarize(
    diagnostic_path: Path,
    mapping_path: Path,
    output_csv: Path,
    output_json: Path,
    iterations: int,
    seed: int,
) -> dict:
    diagnostics = read_jsonl(diagnostic_path)
    mapping = json.loads(mapping_path.read_text())
    if len(diagnostics) != len(mapping):
        raise RuntimeError(
            f"diagnostic/mapping length mismatch: {len(diagnostics)} != {len(mapping)}"
        )
    rows = []
    for item, diagnostic in zip(mapping, diagnostics):
        original = scalar(diagnostic["alignment_score"])
        base = {
            "video_id": item["video_id"],
            "expression_id": str(item["expression_id"]),
            "expression": item.get("question", item.get("expression")),
            "description_type": item.get("description_type", "Sequential"),
            "original_alignment_score": original,
            "beta": float(diagnostic["beta"]),
        }
        permutations = diagnostic.get("permutations")
        if permutations is None:
            legacy = diagnostic.get("permuted_alignment_score")
            permutations = (
                {"reverse": {"alignment_score": legacy}}
                if legacy is not None
                else {}
            )
        for name, value in permutations.items():
            row = {
                **base,
                "permutation": name,
                "permuted_alignment_score": scalar(value["alignment_score"]),
            }
            row["order_preference_margin"] = (
                row["original_alignment_score"] - row["permuted_alignment_score"]
            )
            response = value.get("frame_state_response_l2")
            row["pre_prompt_response_l2"] = (
                float(np.mean(response)) if response is not None else None
            )
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        name: bootstrap(
            [row for row in rows if row["permutation"] == name],
            "order_preference_margin",
            iterations,
            seed,
        )
        for name in sorted({row["permutation"] for row in rows})
    }
    result["beta_mean"] = float(np.mean([row["beta"] for row in rows]))
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.diagnostics,
                args.mapping,
                args.output_csv,
                args.output_json,
                args.bootstrap_iterations,
                args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()

"""Summarize the preregistered CAPG-v2 capability and template controls."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def first_last(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    first, last = {}, {}
    for row in rows:
        first.setdefault(row["question"], row)
        last[row["question"]] = row
    return first, last


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def summarize_one(rows: list[dict], initial_rows: list[dict] | None = None) -> dict:
    first, last = first_last(rows)
    final = list(last.values())
    joint = [row for row in rows if row["stage"] == "joint_sft"]
    by_question: dict[str, list[float]] = {}
    for row in joint:
        by_question.setdefault(row["question"], []).append(row["segmentation_loss"])
    paired = [values[-1] - values[0] for values in by_question.values() if len(values) > 1]
    ordered = [row for row in final if row["verified_order"] and row["order_margin"] is not None]
    objects = [row for row in final if row["object_correct"] is not None]
    envelopes = [row for row in final if row["envelope_tiou"] is not None]
    initial_envelope = None
    if initial_rows is not None:
        _, initial_last = first_last(initial_rows)
        initial_values = [
            row["envelope_tiou"]
            for row in initial_last.values()
            if row["envelope_tiou"] is not None
        ]
        initial_envelope = mean(initial_values)
    length_values = [max(range(6), key=row["length_prior"].__getitem__) + 1 for row in final]
    length_counts = collections.Counter(length_values)
    most_common_n, most_common_count = length_counts.most_common(1)[0]

    def expected_location(row, key):
        probability = np.asarray(row[key], dtype=np.float64)
        denominator = probability.sum()
        positions = np.linspace(0.0, 1.0, len(probability))
        return float((positions * probability).sum() / max(denominator, 1e-12))

    starts = [expected_location(row, "start_posterior") for row in final]
    ends = [expected_location(row, "end_posterior") for row in final]
    posterior = np.asarray([row["posterior_process"] for row in final], dtype=np.float64)
    compatibility = np.asarray([row["compatibility"] for row in final], dtype=np.float64)
    envelope_final = mean(row["envelope_tiou"] for row in envelopes)
    envelope_gain = (
        envelope_final - initial_envelope
        if envelope_final is not None and initial_envelope is not None
        else None
    )
    return {
        "records": len(rows),
        "unique_samples": len(final),
        "paired_segmentation_loss_change": mean(paired),
        "paired_segmentation_decrease_rate": mean(value < 0 for value in paired),
        "original_gt_permutation_rate": mean(row["order_margin"] > 0 for row in ordered),
        "order_samples": len(ordered),
        "mean_order_margin": mean(row["order_margin"] for row in ordered),
        "target_gt_distractor_rate": mean(row["object_correct"] for row in objects),
        "object_samples": len(objects),
        "mean_target_distractor_margin": mean(row["target_distractor_margin"] for row in objects),
        "envelope_tiou_initial": initial_envelope,
        "envelope_tiou_final": envelope_final,
        "envelope_tiou_gain": envelope_gain,
        "envelope_samples": len(envelopes),
        "process_length_distribution": {str(key): value for key, value in sorted(length_counts.items())},
        "distinct_process_lengths": len(length_counts),
        "most_common_n": most_common_n,
        "most_common_n_fraction": most_common_count / len(final),
        "expected_process_length": mean(row["expected_process_length"] for row in final),
        "start_location_std": float(np.std(starts)),
        "end_location_std": float(np.std(ends)),
        "posterior_cross_expression_std": float(posterior.std(axis=0).mean()),
        "compatibility_cross_expression_std": float(compatibility.std(axis=0).mean()),
        "segment_semantic_margin": mean(row["segment_semantic_margin"] for row in final),
        "posterior_entropy": mean(row["posterior_entropy"] for row in final),
        "beta": mean(row["beta"] for row in final),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--global-control", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = read(args.initial)
    adaptive = summarize_one(read(args.adaptive), initial)
    global_control = summarize_one(read(args.global_control), initial)
    gates = {
        "task_fitting": adaptive["paired_segmentation_loss_change"] < 0,
        "order_rate_at_least_90pct": adaptive["original_gt_permutation_rate"] >= 0.90,
        "referent_rate_at_least_90pct": adaptive["target_gt_distractor_rate"] >= 0.90,
        "envelope_tiou_or_gain": (
            adaptive["envelope_tiou_final"] >= 0.50
            or adaptive["envelope_tiou_gain"] >= 0.20
        ),
        "query_dependent_length": (
            adaptive["distinct_process_lengths"] >= 2
            and adaptive["most_common_n_fraction"] <= 0.90
        ),
        "posterior_less_template_like_than_v1": (
            adaptive["posterior_cross_expression_std"] > 0.0171
        ),
        "positive_segment_semantic_margin": adaptive["segment_semantic_margin"] > 0,
    }
    value = {
        "adaptive": adaptive,
        "global_transition_control": global_control,
        "adaptive_minus_global": {
            key: adaptive[key] - global_control[key]
            for key in (
                "envelope_tiou_final",
                "posterior_cross_expression_std",
                "segment_semantic_margin",
                "original_gt_permutation_rate",
                "target_gt_distractor_rate",
            )
        },
        "cpg_v1_posterior_std": 0.0171,
        "gates": gates,
        "capability_gate": "PASS" if all(gates.values()) else "FAIL",
        "pilot_authorized": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

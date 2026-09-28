"""Compile the completed mechanism diagnostics into lightweight tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.temporal_grounding_mechanism.common import (
    object_weighted,
    read_jsonl,
    write_csv,
)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def latest_success(paths: list[Path]) -> list[dict]:
    values = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if row.get("status") == "success":
                values[(row["identity"], row["condition"])] = row
    return list(values.values())


def grouped_means(rows: list[dict], conditions: list[str]) -> list[dict]:
    output = []
    for condition in conditions:
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["description_type"] == description_type
            ]
            objects = object_weighted(selected, ["J", "F", "J_and_F"])
            output.append(
                {
                    "condition": condition,
                    "description_type": description_type,
                    "expressions": len(selected),
                    "objects": len(objects),
                    "videos": len({row["video_id"] for row in objects}),
                    **{
                        field: float(np.mean([row[field] for row in objects]))
                        if objects
                        else None
                        for field in ("J", "F", "J_and_F")
                    },
                }
            )
    return output


def probe_summary(rows: list[dict]) -> list[dict]:
    output = []
    for description_type in ("static", "dynamic", "hybrid"):
        selected = [row for row in rows if row["description_type"] == description_type]
        hits = [row for row in selected if int(row["candidate_hit"])]
        for method in ("static", "temporal"):
            hit_objects = object_weighted(
                hits, [f"{method}_selection_correct"]
            )
            accuracy = float(
                np.mean(
                    [row[f"{method}_selection_correct"] for row in hit_objects]
                )
            ) if hit_objects else None
            objects = object_weighted(
                selected, [f"{method}_candidate_direct_J_and_F"]
            )
            output.append(
                {
                    "description_type": description_type,
                    "method": method,
                    "expressions": len(selected),
                    "candidate_hit_expressions": len(hits),
                    "candidate_miss_expressions": len(selected) - len(hits),
                    "selection_accuracy_on_candidate_hits": accuracy,
                    "candidate_direct_J_and_F": float(
                        np.mean(
                            [
                                row[f"{method}_candidate_direct_J_and_F"]
                                for row in objects
                            ]
                        )
                    )
                    if objects
                    else None,
                    "objects": len(objects),
                }
            )
    return output


def ranking_summary(rows: list[dict]) -> list[dict]:
    output = []
    for budget in (8, 16, 32):
        selected = [row for row in rows if int(row["frame_budget"]) == budget]
        output.append(
            {
                "frame_budget": budget,
                "expressions": len(selected),
                "top1_accuracy": float(
                    np.mean([int(row["top1_is_correct"]) for row in selected])
                ),
                "mean_correct_rank": float(
                    np.mean(
                        [
                            float(row["correct_candidate_rank"])
                            for row in selected
                            if row["correct_candidate_rank"] not in ("", None)
                        ]
                    )
                ),
                "mean_correct_score": float(
                    np.mean([float(row["correct_candidate_score"]) for row in selected])
                ),
                "mean_target_margin": float(
                    np.mean(
                        [float(row["target_vs_best_distractor_margin"]) for row in selected]
                    )
                ),
                "mean_segmentation_J_and_F": float(
                    np.mean([float(row["segmentation_J_and_F"]) for row in selected])
                ),
            }
        )
    return output


def plots(output: Path, ladder: list[dict], probe: list[dict], ranking: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dynamic = {row["condition"]: row for row in ladder if row["description_type"] == "dynamic"}
    order = [
        "ORACLE_CANDIDATE",
        "PREDICTED_CANDIDATE_DIRECT_STATIC",
        "PREDICTED_CANDIDATE_DIRECT_TEMPORAL",
        "PREDICTED_CANDIDATE_POINT_STATIC",
        "PREDICTED_CANDIDATE_POINT_TEMPORAL",
        "ORACLE_ID_POINT",
    ]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(range(len(order)), [dynamic[x]["J_and_F"] for x in order])
    axis.set_xticks(range(len(order)), [x.replace("PREDICTED_CANDIDATE_", "").replace("_", "\n") for x in order])
    axis.set_ylabel("Dynamic J&F")
    axis.set_ylim(0, 1)
    axis.set_title("Oracle ladder (object-weighted)")
    fig.tight_layout()
    fig.savefig(output / "loss_decomposition.png", dpi=180)
    plt.close(fig)

    dynamic_probe = [row for row in probe if row["description_type"] == "dynamic"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    labels = [row["method"].title() for row in dynamic_probe]
    axes[0].bar(labels, [row["selection_accuracy_on_candidate_hits"] for row in dynamic_probe])
    axes[0].set_title("Selection accuracy")
    axes[1].bar(labels, [row["candidate_direct_J_and_F"] for row in dynamic_probe])
    axes[1].set_title("Candidate-direct J&F")
    for axis in axes:
        axis.set_ylim(0, 1)
    fig.suptitle("Frozen diagnostic probe — Dynamic")
    fig.tight_layout()
    fig.savefig(output / "static_vs_temporal_probe.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [row["frame_budget"] for row in ranking],
        [row["mean_target_margin"] for row in ranking],
        marker="o",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Sa2VA visible frames")
    axis.set_ylabel("target − best distractor score")
    axis.set_title("Dynamic candidate margin")
    fig.tight_layout()
    fig.savefig(output / "candidate_margin_vs_frames.png", dpi=180)
    plt.close(fig)


def run(args) -> None:
    root = Path(args.artifact_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    offline = read_csv(root / "oracle_ladder_offline.csv")
    point_paths = sorted(root.glob("point_shard_*/point_results.jsonl"))
    point_attempts = [row for path in point_paths for row in read_jsonl(path)]
    point = latest_success(point_paths)
    successful_point_keys = {
        (row["identity"], row["condition"]) for row in point
    }
    failed_point_attempts = [
        row for row in point_attempts if row.get("status") == "failed"
    ]
    failed_point_keys = {
        (row["identity"], row["condition"]) for row in failed_point_attempts
    }
    reused_prior_tgi = sum(bool(row.get("reused_prior_tgi")) for row in point)
    exact_plan_cache_hits = sum(
        bool(row.get("cache_hit")) and not bool(row.get("reused_prior_tgi"))
        for row in point
    )
    expected_point = len(offline)
    if len(point) != expected_point:
        raise RuntimeError(
            f"point ladder incomplete: {len(point)}/{expected_point} successful rows"
        )
    ladder_rows = offline + point
    conditions = [
        "ORACLE_CANDIDATE",
        "PREDICTED_CANDIDATE_DIRECT_STATIC",
        "PREDICTED_CANDIDATE_DIRECT_TEMPORAL",
        "PREDICTED_CANDIDATE_POINT_STATIC",
        "PREDICTED_CANDIDATE_POINT_TEMPORAL",
        "ORACLE_ID_POINT",
    ]
    counts = defaultdict(int)
    for row in ladder_rows:
        counts[row["condition"]] += 1
    if any(counts[condition] != len(offline) // 3 for condition in conditions):
        raise RuntimeError(f"incomplete condition matrix: {dict(counts)}")
    public_fields = (
        "identity",
        "dataset",
        "video_id",
        "object_id",
        "expression_id",
        "description_type",
        "expression",
        "condition",
        "ORACLE",
        "J",
        "F",
        "J_and_F",
        "selected_track_ids",
        "prompt_updates",
        "cache_hit",
        "reused_prior_tgi",
        "reused_tracking_signature",
        "reused_source_identity",
        "reused_source_condition",
        "latency_seconds_synchronized",
        "peak_memory_bytes",
        "status",
        "gt_entered_model",
    )
    ladder_rows = [
        {field: row.get(field) for field in public_fields} for row in ladder_rows
    ]
    ladder_rows.sort(key=lambda row: (row["identity"], row["condition"]))
    write_csv(output / "oracle_ladder.csv", ladder_rows)
    ladder = grouped_means(ladder_rows, conditions)

    probe_rows = read_csv(root / "probe" / "probe_results.csv")
    probe = probe_summary(probe_rows)
    ranking_rows = read_csv(root / "ranking_diagnosis.csv")
    ranking = ranking_summary(ranking_rows)
    audit = json.loads((root / "probe" / "probe_audit.json").read_text())
    correlations = json.loads((root / "ranking_correlations.json").read_text())
    summary = {
        "artifact_root": str(root.resolve()),
        "oracle_ladder": ladder,
        "probe": probe,
        "probe_bootstrap": {
            "selection_accuracy": audit["dynamic_bootstrap_selection_accuracy"],
            "candidate_direct_J_and_F": audit[
                "dynamic_bootstrap_candidate_direct_J_and_F"
            ],
        },
        "ranking": ranking,
        "ranking_correlations": correlations,
        "completeness": {
            "expressions": len(offline) // 3,
            "oracle_offline_rows": len(offline),
            "point_rows": len(point),
            "probe_rows": len(probe_rows),
            "point_attempt_records": len(point_attempts),
            "point_failure_attempts_retained": len(failed_point_attempts),
            "point_failed_keys_eventually_recovered": len(
                failed_point_keys & successful_point_keys
            ),
            "point_unrecovered_failed_keys": len(
                failed_point_keys - successful_point_keys
            ),
            "point_rows_reused_from_exact_prior_tgi_signature": reused_prior_tgi,
            "point_rows_reused_within_current_exact_plan_cache": exact_plan_cache_hits,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(output, ladder, probe, ranking)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

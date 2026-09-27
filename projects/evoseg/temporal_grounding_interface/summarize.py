"""Summarize a complete pilot/full dynamic-grounding run and apply its gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = [
    "single_global",
    "static_update_k2",
    "static_update_k4",
    "static_update_k8",
    "temporal_update_k2",
    "temporal_update_k4",
    "temporal_update_k8",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(input_roots: list[Path], filename: str) -> list[dict]:
    rows = []
    for root in input_roots:
        path = root / filename
        if path.is_file():
            rows.extend(read_jsonl(path))
    return rows


def cluster_bootstrap(
    values: list[dict], value_field: str, seed: int = 42, iterations: int = 2000
) -> dict:
    by_video = defaultdict(list)
    for value in values:
        by_video[value["video_id"]].append(float(value[value_field]))
    videos = sorted(by_video)
    if not videos:
        return {"mean": None, "ci_low": None, "ci_high": None, "videos": 0, "objects": 0}
    observed = float(np.mean([v for values in by_video.values() for v in values]))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        estimates.append(float(np.mean([v for video in sampled for v in by_video[video]])))
    return {
        "mean": observed,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "videos": len(videos),
        "objects": len(values),
    }


def object_type_means(success: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in success:
        grouped[
            (row["condition"], row["video_id"], row["object_id"], row["description_type"])
        ].append(row)
    values = []
    for (condition, video, object_id, description_type), rows in sorted(grouped.items()):
        values.append(
            {
                "condition": condition,
                "video_id": video,
                "object_id": object_id,
                "description_type": description_type,
                "expressions": len(rows),
                "J": float(np.mean([row["J"] for row in rows])),
                "F": float(np.mean([row["F"] for row in rows])),
                "J_and_F": float(np.mean([row["J_and_F"] for row in rows])),
            }
        )
    return values


def condition_summary(object_means: list[dict]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in object_means
                if row["condition"] == condition
                and row["description_type"] == description_type
            ]
            if not selected:
                continue
            rows.append(
                {
                    "condition": condition,
                    "description_type": description_type,
                    "objects": len(selected),
                    "videos": len({row["video_id"] for row in selected}),
                    "J": float(np.mean([row["J"] for row in selected])),
                    "F": float(np.mean([row["F"] for row in selected])),
                    "J_and_F": float(np.mean([row["J_and_F"] for row in selected])),
                }
            )
    return rows


def paired_rows(object_means: list[dict]) -> tuple[list[dict], list[dict]]:
    index = {
        (row["condition"], row["video_id"], row["object_id"], row["description_type"]): row
        for row in object_means
    }
    dynamic_static = []
    for condition in CONDITIONS:
        objects = sorted(
            {
                (row["video_id"], row["object_id"])
                for row in object_means
                if row["condition"] == condition
            }
        )
        for video, object_id in objects:
            static = index.get((condition, video, object_id, "static"))
            dynamic = index.get((condition, video, object_id, "dynamic"))
            if static and dynamic:
                dynamic_static.append(
                    {
                        "comparison": "dynamic_minus_static",
                        "condition": condition,
                        "video_id": video,
                        "object_id": object_id,
                        "static_J_and_F": static["J_and_F"],
                        "dynamic_J_and_F": dynamic["J_and_F"],
                        "difference": dynamic["J_and_F"] - static["J_and_F"],
                    }
                )
    temporal_static = []
    for k in (2, 4, 8):
        left = f"static_update_k{k}"
        right = f"temporal_update_k{k}"
        objects = sorted(
            {
                (row["video_id"], row["object_id"])
                for row in object_means
                if row["condition"] == left and row["description_type"] == "dynamic"
            }
        )
        for video, object_id in objects:
            static = index.get((left, video, object_id, "dynamic"))
            temporal = index.get((right, video, object_id, "dynamic"))
            if static and temporal:
                temporal_static.append(
                    {
                        "comparison": "temporal_minus_static_dynamic",
                        "K": k,
                        "video_id": video,
                        "object_id": object_id,
                        "static_update_dynamic_J_and_F": static["J_and_F"],
                        "temporal_update_dynamic_J_and_F": temporal["J_and_F"],
                        "difference": temporal["J_and_F"] - static["J_and_F"],
                    }
                )
    return dynamic_static, temporal_static


def transition_summary(stage_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in stage_rows:
        grouped[(row["condition"], row["identity"])].append(row)
    records = []
    for (condition, identity), values in grouped.items():
        values.sort(key=lambda value: int(value["stage_order"]))
        correct = [bool(value["selection_correct"]) for value in values]
        records.append(
            {
                "condition": condition,
                "identity": identity,
                "video_id": values[0]["video_id"],
                "description_type": values[0]["description_type"],
                "initial_correct": correct[0],
                "initial_wrong_later_corrected": (not correct[0]) and any(correct[1:]),
                "initial_correct_later_damaged": correct[0] and any(not value for value in correct[1:]),
                "final_correct": correct[-1],
            }
        )
    summaries = []
    for condition in CONDITIONS:
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in records
                if row["condition"] == condition
                and row["description_type"] == description_type
            ]
            if not selected:
                continue
            initial_wrong = [row for row in selected if not row["initial_correct"]]
            initial_correct = [row for row in selected if row["initial_correct"]]
            summaries.append(
                {
                    "condition": condition,
                    "description_type": description_type,
                    "expressions": len(selected),
                    "initial_selection_accuracy": float(np.mean([row["initial_correct"] for row in selected])),
                    "initial_wrong_count": len(initial_wrong),
                    "correction_rate_given_initial_wrong": float(
                        np.mean([row["initial_wrong_later_corrected"] for row in initial_wrong])
                    )
                    if initial_wrong
                    else None,
                    "initial_correct_count": len(initial_correct),
                    "damage_rate_given_initial_correct": float(
                        np.mean([row["initial_correct_later_damaged"] for row in initial_correct])
                    )
                    if initial_correct
                    else None,
                    "final_selection_accuracy": float(np.mean([row["final_correct"] for row in selected])),
                }
            )
    return summaries


def plot_results(summary: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    lookup = {(row["condition"], row["description_type"]): row for row in summary}
    labels = CONDITIONS
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5))
    for offset, kind in enumerate(("static", "dynamic", "hybrid")):
        values = [lookup.get((condition, kind), {}).get("J_and_F", np.nan) * 100 for condition in labels]
        axis.bar(x + (offset - 1) * width, values, width, label=kind)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("J&F (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "jf_vs_num_updates.png", dpi=180)
    plt.close(figure)

    gaps = []
    for condition in labels:
        static = lookup.get((condition, "static"))
        dynamic = lookup.get((condition, "dynamic"))
        gaps.append((dynamic["J_and_F"] - static["J_and_F"]) * 100 if static and dynamic else np.nan)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.axhline(0, color="black", linewidth=1)
    axis.bar(x, gaps)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("Dynamic - Static J&F (pp)")
    figure.tight_layout()
    figure.savefig(output / "dynamic_static_gap.png", dpi=180)
    plt.close(figure)


def run(args) -> None:
    roots = [Path(value).resolve() for value in args.input_root]
    docs = Path(args.docs_dir).resolve()
    docs.mkdir(parents=True, exist_ok=True)
    prediction_attempts = collect(roots, "dynamic_predictions.jsonl")
    terminal = {}
    for row in prediction_attempts:
        terminal[(row["identity"], row["condition"])] = row
    predictions = list(terminal.values())
    stage_attempts = collect(roots, "dynamic_stages.jsonl")
    terminal_stages = {}
    for row in stage_attempts:
        terminal_stages[
            (row["identity"], row["condition"], int(row["stage_order"]))
        ] = row
    stages = list(terminal_stages.values())
    successes = [row for row in predictions if row.get("status") == "success"]
    failures = [row for row in predictions if row.get("status") != "success"]
    keys = [(row["identity"], row["condition"]) for row in successes]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate successful expression/condition rows")
    expected_conditions = set(CONDITIONS)
    by_identity = defaultdict(set)
    for row in successes:
        by_identity[row["identity"]].add(row["condition"])
    complete_identities = {
        identity for identity, conditions in by_identity.items() if conditions == expected_conditions
    }
    analysis_success = [row for row in successes if row["identity"] in complete_identities]
    analysis_stages = [row for row in stages if row["identity"] in complete_identities]
    object_means = object_type_means(analysis_success)
    summary = condition_summary(object_means)
    dynamic_static, temporal_static = paired_rows(object_means)
    transitions = transition_summary(analysis_stages)

    comparison_summaries = []
    for condition in CONDITIONS:
        selected = [row for row in dynamic_static if row["condition"] == condition]
        comparison_summaries.append(
            {"comparison": "dynamic_minus_static", "condition": condition, **cluster_bootstrap(selected, "difference")}
        )
    for k in (2, 4, 8):
        selected = [row for row in temporal_static if row["K"] == k]
        comparison_summaries.append(
            {"comparison": "temporal_minus_static_dynamic", "K": k, **cluster_bootstrap(selected, "difference")}
        )

    write_csv(docs / "per_expression_results.csv", predictions)
    write_csv(docs / "per_stage_selection.csv", stages)
    write_csv(docs / "paired_dynamic_results.csv", dynamic_static + temporal_static)
    write_csv(docs / "condition_summary.csv", summary)
    write_csv(docs / "bootstrap_summary.csv", comparison_summaries)
    write_csv(docs / "correction_transition.csv", transitions)
    plot_results(summary, docs)

    summary_lookup = {(row["condition"], row["description_type"]): row for row in summary}
    comparison_lookup = {
        (row["comparison"], row.get("condition", row.get("K"))): row
        for row in comparison_summaries
    }
    transition_lookup = {
        (row["condition"], row["description_type"]): row for row in transitions
    }
    primary = comparison_lookup[("temporal_minus_static_dynamic", 4)]
    signs = [
        comparison_lookup[("temporal_minus_static_dynamic", k)]["mean"] > 0
        for k in (2, 4, 8)
        if comparison_lookup[("temporal_minus_static_dynamic", k)]["mean"] is not None
    ]
    single_gap = comparison_lookup[("dynamic_minus_static", "single_global")]["mean"]
    temporal_gap = comparison_lookup[("dynamic_minus_static", "temporal_update_k4")]["mean"]
    gap_shrink = abs(single_gap) - abs(temporal_gap) if single_gap is not None and temporal_gap is not None else None
    temporal_correction = transition_lookup.get(("temporal_update_k4", "dynamic"), {})
    static_correction = transition_lookup.get(("static_update_k4", "dynamic"), {})
    correction_advantage = (
        temporal_correction.get("correction_rate_given_initial_wrong")
        - static_correction.get("correction_rate_given_initial_wrong")
        if temporal_correction.get("correction_rate_given_initial_wrong") is not None
        and static_correction.get("correction_rate_given_initial_wrong") is not None
        else None
    )
    gate_a = primary["mean"] is not None and primary["mean"] > 0 and sum(signs) >= 2
    gate_b = gap_shrink is not None and gap_shrink >= 0.01
    gate_c = correction_advantage is not None and correction_advantage > 0
    go = bool(gate_a and gate_b and gate_c and not failures)
    decision = {
        "decision": "GO" if go else "NO-GO",
        "primary_comparison": "temporal_update_k4 minus static_update_k4 on Dynamic object means",
        "gate_A_temporal_advantage_and_direction_stability": gate_a,
        "gate_B_dynamic_static_gap_shrinks_by_at_least_1pp": gate_b,
        "gate_C_more_real_referent_corrections": gate_c,
        "primary_effect": primary,
        "positive_K_directions": int(sum(signs)),
        "single_global_dynamic_static_gap": single_gap,
        "temporal_k4_dynamic_static_gap": temporal_gap,
        "absolute_gap_shrink": gap_shrink,
        "temporal_minus_static_correction_rate": correction_advantage,
        "successful_rows": len(successes),
        "failed_rows": len(failures),
        "complete_expression_identities": len(complete_identities),
        "excluded_incomplete_identities": len(by_identity) - len(complete_identities),
    }
    (docs / "gate.json").write_text(json.dumps(decision, indent=2) + "\n")

    def percentage(value):
        return "N/A" if value is None else f"{value * 100:.2f}"

    table = ["| Condition | Type | J | F | J&F |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        table.append(
            f"| {row['condition']} | {row['description_type']} | {row['J']*100:.2f} | {row['F']*100:.2f} | {row['J_and_F']*100:.2f} |"
        )
    report = f"""# Dynamic Grounding Interface results

本页只统计七个条件均成功的完整 expression；失败记录保留且不补值。当前覆盖 {len(complete_identities)} expressions，失败 {len(failures)}。

## 主结果

{chr(10).join(table)}

预注册主比较是 K=4 的 Dynamic object mean：Temporal - Static = {percentage(primary['mean'])} pp，source-video cluster bootstrap 95% CI [{percentage(primary['ci_low'])}, {percentage(primary['ci_high'])}] pp。

Single-global Dynamic-Static gap = {percentage(single_gap)} pp；Temporal K4 gap = {percentage(temporal_gap)} pp；绝对 gap 缩小 {percentage(gap_shrink)} pp。

## Decision gate

- A（K4 Temporal > Static 且三档至少两档方向为正）：{gate_a}
- B（相对 Single-global，gap 至少缩小 1pp）：{gate_b}
- C（Dynamic 初始选错后的真实 correction rate 高于 Static control）：{gate_c}
- **{decision['decision']}**

这只是当前冻结 Sa2VA + 官方 SAM3.1 point refinement、零训练参数 scorer 的结论；不外推为所有动态接口均有效或无效。
"""
    (docs / args.report_name).write_text(report)
    (docs / "FAILURE_ANALYSIS.md").write_text(
        "# Failure analysis\n\n"
        f"运行失败 {len(failures)} 条；不完整 expression {decision['excluded_incomplete_identities']} 条，均未进入主要统计。"
        "`per_stage_selection.csv` 保留初始选错后纠正、初始选对后改坏及每次 update 前后 anchor J&F。\n"
    )
    (docs / "FINAL_GO_NOGO.md").write_text(
        "# Final GO / NO-GO\n\n"
        f"当前决策：**{decision['decision']}**。\n\n"
        "该文件由预注册的 K4 主比较、跨 K 方向稳定性、gap 缩小和真实 referent correction 三项联合 gate 生成；不会选择事后最优 K。\n"
    )
    print(json.dumps(decision, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--report-name", default="PILOT_RESULTS.md")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

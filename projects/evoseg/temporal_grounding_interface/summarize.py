"""Summarize a complete pilot/full dynamic-grounding run and apply its gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


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


def collect(
    input_roots: list[Path], filename: str, *, annotate_root: bool = False
) -> list[dict]:
    rows = []
    for root in input_roots:
        path = root / filename
        if path.is_file():
            values = read_jsonl(path)
            if annotate_root:
                for value in values:
                    value["_source_root"] = str(root)
            rows.extend(values)
    return rows


def mask_activity_summary(success: list[dict]) -> list[dict]:
    """Audit saved predictions rather than inferring activity from DAVIS scores.

    Empty GT frames legitimately score one for an empty prediction, so J/F alone
    cannot detect a degenerate all-empty condition.
    """

    grouped = defaultdict(list)
    for row in success:
        mask_path = Path(row["_source_root"]) / row["prediction_masks_path"]
        payload = json.loads(mask_path.read_text())
        frame_areas = []
        for value in payload:
            rle = dict(value["rle"])
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            frame_areas.append(float(mask_utils.area(rle)))
        grouped[(row["condition"], row["description_type"])].append(frame_areas)
    result = []
    for (condition, description_type), expressions in sorted(grouped.items()):
        all_areas = [area for expression in expressions for area in expression]
        result.append(
            {
                "condition": condition,
                "description_type": description_type,
                "expressions": len(expressions),
                "all_empty_expressions": sum(max(value, default=0.0) == 0 for value in expressions),
                "all_empty_expression_fraction": float(
                    np.mean([max(value, default=0.0) == 0 for value in expressions])
                ),
                "nonempty_evaluation_frame_fraction": float(
                    np.mean([value > 0 for value in all_areas])
                ),
            }
        )
    return result


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


def runtime_summary(success: list[dict]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in success
                if row["condition"] == condition
                and row["description_type"] == description_type
            ]
            if not selected:
                continue
            rows.append(
                {
                    "condition": condition,
                    "description_type": description_type,
                    "expressions": len(selected),
                    "median_total_latency_seconds": float(
                        np.median([row["latency_seconds_synchronized"] for row in selected])
                    ),
                    "median_vlm_latency_seconds": float(
                        np.median([row["vlm_latency_seconds_synchronized"] for row in selected])
                    ),
                    "median_candidate_latency_seconds": float(
                        np.median(
                            [row["candidate_latency_seconds_synchronized"] for row in selected]
                        )
                    ),
                    "median_sam31_latency_seconds": float(
                        np.median([row["sam31_latency_seconds_synchronized"] for row in selected])
                    ),
                    "max_peak_memory_gib": float(
                        max(row["peak_memory_bytes"] for row in selected) / 2**30
                    ),
                    "mean_prompt_updates": float(
                        np.mean([row["prompt_updates"] for row in selected])
                    ),
                    "mean_vlm_forward_count": float(
                        np.mean([row["vlm_forward_count"] for row in selected])
                    ),
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


def transition_summary(stage_rows: list[dict]) -> tuple[list[dict], list[dict]]:
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
    return summaries, records


def plot_results(
    summary: list[dict], stage_rows: list[dict], transitions: list[dict], output: Path
) -> None:
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

    figure, axis = plt.subplots(figsize=(8, 4))
    for condition in ("static_update_k8", "temporal_update_k8"):
        selected = [
            row
            for row in stage_rows
            if row["condition"] == condition and row["description_type"] == "dynamic"
        ]
        by_stage = defaultdict(list)
        for row in selected:
            by_stage[int(row["stage_order"])].append(float(row["selection_correct"]))
        axis.plot(
            sorted(by_stage),
            [np.mean(by_stage[index]) * 100 for index in sorted(by_stage)],
            marker="o",
            label=condition,
        )
    axis.set_xlabel("Stage")
    axis.set_ylabel("Dynamic selection accuracy (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "stage_selection_accuracy.png", dpi=180)
    plt.close(figure)

    transition_lookup = {
        (row["condition"], row["description_type"]): row for row in transitions
    }
    labels = ["K=2", "K=4", "K=8"]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(8, 4))
    for offset, prefix in enumerate(("static_update", "temporal_update")):
        values = []
        for k in (2, 4, 8):
            value = transition_lookup.get((f"{prefix}_k{k}", "dynamic"), {})
            rate = value.get("correction_rate_given_initial_wrong")
            values.append(np.nan if rate is None else rate * 100)
        axis.bar(x + (offset - 0.5) * 0.35, values, 0.35, label=prefix)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Correction rate given initial error (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "correction_transition.png", dpi=180)
    plt.close(figure)


def run(args) -> None:
    roots = [Path(value).resolve() for value in args.input_root]
    docs = Path(args.docs_dir).resolve()
    docs.mkdir(parents=True, exist_ok=True)
    prediction_attempts = collect(
        roots, "dynamic_predictions.jsonl", annotate_root=True
    )
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
    runtimes = runtime_summary(analysis_success)
    mask_activity = mask_activity_summary(analysis_success)
    dynamic_static, temporal_static = paired_rows(object_means)
    transitions, transition_details = transition_summary(analysis_stages)

    stage_index = {
        (row["identity"], row["condition"], int(row["stage_order"])): row
        for row in analysis_stages
    }
    result_index = {
        (row["identity"], row["condition"]): row for row in analysis_success
    }
    interface_diagnostics = []
    for identity in sorted(complete_identities):
        description_type = result_index[(identity, "single_global")]["description_type"]
        video_id = result_index[(identity, "single_global")]["video_id"]
        for k in (2, 4, 8):
            static_condition = f"static_update_k{k}"
            temporal_condition = f"temporal_update_k{k}"
            static_selected = [
                stage_index[(identity, static_condition, order)]["selected_candidate_index"]
                for order in range(k)
            ]
            temporal_selected = [
                stage_index[(identity, temporal_condition, order)]["selected_candidate_index"]
                for order in range(k)
            ]
            differing = sum(
                left != right for left, right in zip(static_selected, temporal_selected)
            )
            interface_diagnostics.append(
                {
                    "identity": identity,
                    "video_id": video_id,
                    "description_type": description_type,
                    "K": k,
                    "static_selected_candidates": json.dumps(static_selected),
                    "temporal_selected_candidates": json.dumps(temporal_selected),
                    "differing_stage_count": differing,
                    "any_selection_difference": bool(differing),
                    "temporal_minus_static_J_and_F": result_index[
                        (identity, temporal_condition)
                    ]["J_and_F"]
                    - result_index[(identity, static_condition)]["J_and_F"],
                }
            )

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

    public_predictions = [
        {key: value for key, value in row.items() if key != "_source_root"}
        for row in predictions
    ]
    write_csv(docs / "per_expression_results.csv", public_predictions)
    write_csv(docs / "per_stage_selection.csv", stages)
    write_csv(docs / "paired_dynamic_results.csv", dynamic_static + temporal_static)
    write_csv(docs / "condition_summary.csv", summary)
    write_csv(docs / "runtime_summary.csv", runtimes)
    write_csv(docs / "mask_activity_summary.csv", mask_activity)
    write_csv(docs / "bootstrap_summary.csv", comparison_summaries)
    write_csv(docs / "correction_transition.csv", transitions)
    write_csv(docs / "correction_transition_details.csv", transition_details)
    write_csv(docs / "interface_diagnostics.csv", interface_diagnostics)
    plot_results(summary, analysis_stages, transitions, docs)

    summary_lookup = {(row["condition"], row["description_type"]): row for row in summary}
    runtime_lookup = {
        (row["condition"], row["description_type"]): row for row in runtimes
    }
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
    prediction_index = {
        (row["identity"], row["condition"]): row for row in analysis_success
    }
    corrected_dynamic_identities = [
        row["identity"]
        for row in transition_details
        if row["condition"] == "temporal_update_k4"
        and row["description_type"] == "dynamic"
        and row["initial_wrong_later_corrected"]
    ]
    correction_linked_gains = [
        prediction_index[(identity, "temporal_update_k4")]["J_and_F"]
        - prediction_index[(identity, "static_update_k4")]["J_and_F"]
        for identity in corrected_dynamic_identities
        if (identity, "temporal_update_k4") in prediction_index
        and (identity, "static_update_k4") in prediction_index
    ]
    correction_linked_mean_gain = (
        float(np.mean(correction_linked_gains)) if correction_linked_gains else None
    )
    dynamic_k4_diagnostics = [
        row
        for row in interface_diagnostics
        if row["description_type"] == "dynamic" and row["K"] == 4
    ]
    dynamic_k4_selection_divergence = (
        float(np.mean([row["any_selection_difference"] for row in dynamic_k4_diagnostics]))
        if dynamic_k4_diagnostics
        else None
    )
    gate_a = primary["mean"] is not None and primary["mean"] > 0 and sum(signs) >= 2
    gate_b = gap_shrink is not None and gap_shrink >= 0.01
    gate_c = (
        correction_advantage is not None
        and correction_advantage > 0
        and correction_linked_mean_gain is not None
        and correction_linked_mean_gain > 0
    )
    single_activity = [
        row for row in mask_activity if row["condition"] == "single_global"
    ]
    single_global_protocol_valid = bool(single_activity) and any(
        row["all_empty_expression_fraction"] < 1.0 for row in single_activity
    )
    go = bool(
        gate_a
        and gate_b
        and gate_c
        and single_global_protocol_valid
        and not failures
    )
    decision = {
        "decision": "GO" if go else "NO-GO",
        "primary_comparison": "temporal_update_k4 minus static_update_k4 on Dynamic object means",
        "gate_A_temporal_advantage_and_direction_stability": gate_a,
        "gate_B_dynamic_static_gap_shrinks_by_at_least_1pp": gate_b,
        "gate_C_more_real_referent_corrections": gate_c,
        "single_global_protocol_valid": single_global_protocol_valid,
        "primary_effect": primary,
        "positive_K_directions": int(sum(signs)),
        "single_global_dynamic_static_gap": single_gap,
        "temporal_k4_dynamic_static_gap": temporal_gap,
        "absolute_gap_shrink": gap_shrink,
        "temporal_minus_static_correction_rate": correction_advantage,
        "corrected_dynamic_expression_count": len(correction_linked_gains),
        "mean_temporal_minus_static_JF_on_corrected_dynamic_expressions": correction_linked_mean_gain,
        "dynamic_k4_expression_fraction_with_different_candidate_sequence": dynamic_k4_selection_divergence,
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
    latency_table = [
        "| Condition | Dynamic median total (s) | VLM forwards | Updates | Peak GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = runtime_lookup.get((condition, "dynamic"))
        if row:
            latency_table.append(
                f"| {condition} | {row['median_total_latency_seconds']:.2f} | "
                f"{row['mean_vlm_forward_count']:.2f} | {row['mean_prompt_updates']:.2f} | "
                f"{row['max_peak_memory_gib']:.2f} |"
            )
    covered_objects = len(
        {
            (row["video_id"], row["object_id"])
            for row in analysis_success
        }
    )
    covered_videos = len({row["video_id"] for row in analysis_success})
    report = f"""# Dynamic Grounding Interface results

本页只统计七个条件均成功的完整 expression；失败记录保留且不补值。当前覆盖 {covered_videos} videos、{covered_objects} objects、{len(complete_identities)} expressions，terminal 失败 {len(failures)}。

## 主结果

{chr(10).join(table)}

预注册主比较是 K=4 的 Dynamic object mean：Temporal - Static = {percentage(primary['mean'])} pp，source-video cluster bootstrap 95% CI [{percentage(primary['ci_low'])}, {percentage(primary['ci_high'])}] pp。

Single-global Dynamic-Static gap = {percentage(single_gap)} pp；Temporal K4 gap = {percentage(temporal_gap)} pp；绝对 gap 缩小 {percentage(gap_shrink)} pp。

K4 Dynamic 中，Temporal 与 Static 产生不同 candidate sequence 的 expression 比例为 {percentage(dynamic_k4_selection_divergence)}%。该诊断区分“表示变化没有越过离散 grounding 决策边界”和“改变了 referent 但 tracking/掩码未改善”。

## 运行代价

{chr(10).join(latency_table)}

延迟均包含 CUDA synchronize；tracking cache hit 仍报告首次独立执行该 trajectory 的同步延迟。VLM/candidate 缓存来自前置独立运行，且部分 GPU 与既有任务并行，因此这些数字用于如实记录本轮代价，不作为隔离效率结论。

## Decision gate

- A（K4 Temporal > Static 且三档至少两档方向为正）：{gate_a}
- B（相对 Single-global，gap 至少缩小 1pp）：{gate_b}
- C（Dynamic correction rate 高于 Static control，且被纠正样本的 Temporal-Static J&F 为正）：{gate_c}；被纠正 {len(correction_linked_gains)} 条，平均关联增益 {percentage(correction_linked_mean_gain)} pp
- Single-global 输出活动性检查：{single_global_protocol_valid}（逐条件明细见 `mask_activity_summary.csv`；全空 mask 不会因 GT 缺席帧的 J/F=1 被误判为有效输出）
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

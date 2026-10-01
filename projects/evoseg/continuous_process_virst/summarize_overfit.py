"""Summarize the preregistered CPG capability gate without result tuning."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def last_by_question(rows: list[dict]) -> list[dict]:
    values = {}
    for row in rows:
        values[row["question"]] = row
    return list(values.values())


def summarize(rows: list[dict], planned: int) -> dict:
    joint = [row for row in rows if row["stage"] == "joint_sft"]
    window = min(32, max(1, len(joint) // 4)) if joint else 0
    early_seg = mean([row["segmentation_loss"] for row in joint[:window]])
    late_seg = mean([row["segmentation_loss"] for row in joint[-window:]])
    seg_change = late_seg - early_seg if early_seg is not None and late_seg is not None else None
    per_question: dict[str, list[float]] = {}
    for row in joint:
        per_question.setdefault(row["question"], []).append(row["segmentation_loss"])
    paired_changes = [
        values[-1] - values[0] for values in per_question.values() if len(values) >= 2
    ]
    paired_seg_change = mean(paired_changes)
    paired_seg_decrease_rate = mean([float(value < 0) for value in paired_changes])
    last = last_by_question(rows)
    ordered = [row for row in last if row["verified_order"] and row["order_margin"] is not None]
    objects = [row for row in last if row["object_correct"] is not None]
    order_rate = mean([float(row["order_margin"] > 0) for row in ordered])
    object_rate = mean([float(row["object_correct"]) for row in objects])
    expected_length = mean([row["expected_process_length"] for row in last])
    duration = mean([row["mean_state_duration"] for row in last])
    entropy = mean([row["posterior_entropy"] for row in last])
    posterior_rows = [row for row in last if row.get("process_posterior")]
    monotonic_values = []
    noncollapse_values = []
    for row in posterior_rows:
        posterior = row["process_posterior"]
        expected_state = [
            sum((state + 1) * probability for state, probability in enumerate(frame))
            for frame in posterior
        ]
        monotonic_values.append(
            float(all(right + 1e-5 >= left for left, right in zip(expected_state, expected_state[1:])))
        )
        duration = [sum(frame[state] for frame in posterior) for state in range(len(posterior[0]))]
        noncollapse_values.append(float(max(duration) / len(posterior) < 0.90))
    posterior_monotonic_rate = mean(monotonic_values)
    posterior_noncollapse_rate = mean(noncollapse_values)
    gates = {
        "segmentation_loss_decreased": (
            paired_seg_change is not None and paired_seg_change < 0
        ),
        "interval_localization_improved": False,
        "original_gt_reverse_rate_at_least_90pct": order_rate is not None and order_rate >= 0.90,
        "target_gt_distractor_rate_at_least_85pct": object_rate is not None and object_rate >= 0.85,
        "posterior_non_degenerate": (
            expected_length is not None
            and entropy is not None
            and posterior_monotonic_rate is not None
            and posterior_noncollapse_rate is not None
            and expected_length > 1.5
            and 0.05 < entropy < math.log(6.0) - 0.01
            and posterior_monotonic_rate >= 0.95
            and posterior_noncollapse_rate >= 0.90
        ),
    }
    complete = len(rows) == planned
    return {
        "planned_updates": planned,
        "completed_updates": len(rows),
        "complete": complete,
        "segmentation_loss_early": early_seg,
        "segmentation_loss_late": late_seg,
        "segmentation_loss_change": seg_change,
        "segmentation_paired_questions": len(paired_changes),
        "segmentation_paired_mean_change": paired_seg_change,
        "segmentation_paired_decrease_rate": paired_seg_decrease_rate,
        "order_unique_samples": len(ordered),
        "original_gt_reverse_rate": order_rate,
        "object_unique_samples": len(objects),
        "target_gt_distractor_rate": object_rate,
        "interval_localization": None,
        "interval_supervision": "unavailable: official interval is process-span, not clause-span",
        "expected_process_length": expected_length,
        "mean_state_duration": duration,
        "posterior_entropy": entropy,
        "posterior_samples": len(posterior_rows),
        "posterior_monotonic_rate": posterior_monotonic_rate,
        "posterior_noncollapse_rate": posterior_noncollapse_rate,
        "gates": gates,
        "overfit_gate": "PASS" if complete and all(gates.values()) else "FAIL",
        "pilot_authorized": complete and all(gates.values()),
    }


def report(value: dict) -> str:
    gates = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in value["gates"].items()
    )
    return f"""# CPG-VIRST Overfit Capability Gate

Status: **{value['overfit_gate']}** ({value['completed_updates']}/{value['planned_updates']} updates)

## Measurements

- Segmentation loss, early → late: `{value['segmentation_loss_early']}` → `{value['segmentation_loss_late']}` (delta `{value['segmentation_loss_change']}`)
- Per-question first → last segmentation loss: mean delta `{value['segmentation_paired_mean_change']}`, decrease rate `{value['segmentation_paired_decrease_rate']}` over `{value['segmentation_paired_questions']}` repeated questions
- Original > fixed permutation rate: `{value['original_gt_reverse_rate']}` over `{value['order_unique_samples']}` unique verified-order samples
- Target > best distractor rate: `{value['target_gt_distractor_rate']}` over `{value['object_unique_samples']}` unique samples with official distractor masks
- Interval localization: `N/A`; GroundMoRe's official interval is the whole queried-process/mask-validity span, not an unambiguous target-clause label
- Expected process length: `{value['expected_process_length']}`
- Mean state duration: `{value['mean_state_duration']}` frames
- Posterior entropy: `{value['posterior_entropy']}`
- Posterior expected-state monotonic rate: `{value['posterior_monotonic_rate']}`
- Posterior non-single-state-collapse rate: `{value['posterior_noncollapse_rate']}`

## Preregistered checks

{gates}

Pilot launch authorized: **{value['pilot_authorized']}**.

The unavailable interval label is not replaced by a pseudo-label. Consequently
the full preregistered capability gate cannot pass under the current official
metadata, even if the other training checks succeed.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--planned", type=int, default=576)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.training.read_text().splitlines() if line]
    value = summarize(rows, args.planned)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(value, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(value))
    print(json.dumps(value))


if __name__ == "__main__":
    main()

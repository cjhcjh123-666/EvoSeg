"""Aggregate controlled FTG/Frame Prompt pairs across optimization seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


GROUPS = (
    "long_rvos/static",
    "long_rvos/dynamic",
    "long_rvos/hybrid",
    "mevis_v2",
    "overall",
)


def _load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete result: {path}")
    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def summarize_pairs(
    pairs: list[tuple[str, Path, Path]],
    selective_gain: float = 0.01,
    static_floor: float = -0.005,
) -> dict:
    if not pairs:
        raise ValueError("at least one controlled pair is required")
    records = []
    for label, frame_path, ftg_path in pairs:
        frame = _load(frame_path)
        ftg = _load(ftg_path)
        if frame["variant"] != "frame_prompt" or ftg["variant"] != "ftg":
            raise RuntimeError(f"{label}: expected Frame Prompt and FTG results")
        if frame["manifest_sha256"] != ftg["manifest_sha256"]:
            raise RuntimeError(f"{label}: controlled pair used different manifests")
        frame_groups = frame["validation"]["original"]
        ftg_groups = ftg["validation"]["original"]
        group_values = {}
        for group in GROUPS:
            frame_value = frame_groups[group]["jf"]
            ftg_value = ftg_groups[group]["jf"]
            group_values[group] = {
                "frame": frame_value,
                "ftg": ftg_value,
                "delta": ftg_value - frame_value,
                "count": ftg_groups[group]["count"],
            }
        records.append(
            {
                "label": label,
                "manifest_sha256": frame["manifest_sha256"],
                "groups": group_values,
                "overall_oracle_delta": (
                    ftg_groups["overall"]["oracle_jf"]
                    - frame_groups["overall"]["oracle_jf"]
                ),
                "query_accuracy_delta": (
                    ftg_groups["overall"]["query_selection_accuracy"]
                    - frame_groups["overall"]["query_selection_accuracy"]
                ),
                "predicted_switch_delta": (
                    ftg_groups["overall"]["predicted_query_switch_rate"]
                    - frame_groups["overall"]["predicted_query_switch_rate"]
                ),
            }
        )

    aggregate = {}
    for group in GROUPS:
        frame_values = [record["groups"][group]["frame"] for record in records]
        ftg_values = [record["groups"][group]["ftg"] for record in records]
        deltas = [record["groups"][group]["delta"] for record in records]
        aggregate[group] = {
            "frame": _mean_std(frame_values),
            "ftg": _mean_std(ftg_values),
            "delta": _mean_std(deltas),
            "positive_pairs": sum(value > 0 for value in deltas),
            "pair_count": len(deltas),
            "validation_count_per_pair": records[0]["groups"][group]["count"],
        }

    dynamic_mean = statistics.mean(
        aggregate[group]["delta"]["mean"]
        for group in ("long_rvos/dynamic", "long_rvos/hybrid", "mevis_v2")
    )
    static_delta = aggregate["long_rvos/static"]["delta"]["mean"]
    direction_consistent = all(
        aggregate[group]["positive_pairs"] == len(records)
        for group in ("long_rvos/dynamic", "long_rvos/hybrid", "mevis_v2")
    )
    passed = dynamic_mean >= selective_gain and static_delta >= static_floor
    summary = {
        "comparison": "ftg_minus_frame_prompt",
        "pair_count": len(records),
        "pairs": records,
        "aggregate": aggregate,
        "dynamic_motion_mean_delta": dynamic_mean,
        "static_delta": static_delta,
        "direction_consistent_across_pairs": direction_consistent,
        "selective_over_static": dynamic_mean > static_delta,
        "decision": "GO" if passed else "NO-GO",
        "claim_status": (
            "SUPPORTED"
            if passed and direction_consistent and dynamic_mean > static_delta
            else "MIXED"
        ),
        "thresholds": {
            "dynamic_mean_minimum": selective_gain,
            "static_minimum": static_floor,
        },
    }
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "| Group | Frame mean±std | FTG mean±std | Δ mean±std | Positive seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        values = summary["aggregate"][group]
        lines.append(
            "| {group} | {fm:.2f}±{fs:.2f} | {tm:.2f}±{ts:.2f} | "
            "{dm:+.2f}±{ds:.2f} | {positive}/{count} |".format(
                group=group,
                fm=100 * values["frame"]["mean"],
                fs=100 * values["frame"]["std"],
                tm=100 * values["ftg"]["mean"],
                ts=100 * values["ftg"]["std"],
                dm=100 * values["delta"]["mean"],
                ds=100 * values["delta"]["std"],
                positive=values["positive_pairs"],
                count=values["pair_count"],
            )
        )
    lines.extend(
        [
            "",
            f"Decision: **{summary['decision']}**; claim: **{summary['claim_status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True,
        metavar=("LABEL", "FRAME_RESULT", "FTG_RESULT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selective-gain", type=float, default=0.01)
    parser.add_argument("--static-floor", type=float, default=-0.005)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controlled_pairs = [
        (label, Path(frame), Path(ftg)) for label, frame, ftg in args.pair
    ]
    result = summarize_pairs(
        controlled_pairs, args.selective_gain, args.static_floor
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replication_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output / "replication_summary.md").write_text(render_markdown(result))
    print(json.dumps(result, indent=2))

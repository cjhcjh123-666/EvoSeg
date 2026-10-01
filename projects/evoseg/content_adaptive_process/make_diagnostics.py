"""Export lightweight final-epoch CAPG capability diagnostics and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_final(path: Path) -> list[dict]:
    final = {}
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            final[row["question"]] = row
    return list(final.values())


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def expected_location(values: list[float]) -> float:
    probability = np.asarray(values, dtype=np.float64)
    positions = np.linspace(0, 1, len(probability))
    return float((probability * positions).sum() / max(probability.sum(), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--global-control", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--posterior-figure", type=Path, required=True)
    args = parser.parse_args()
    adaptive = read_final(args.adaptive)
    global_control = {row["question"]: row for row in read_final(args.global_control)}
    records = []
    for row in adaptive:
        control = global_control[row["question"]]
        records.append(
            {
                "question": row["question"],
                "expression_id": row["expression_id"],
                "argmax_N": max(range(6), key=row["length_prior"].__getitem__) + 1,
                "expected_N": row["expected_process_length"],
                "start_location": expected_location(row["start_posterior"]),
                "end_location": expected_location(row["end_posterior"]),
                "envelope_tIoU": row["envelope_tiou"],
                "order_margin": row["order_margin"],
                "object_margin": row["target_distractor_margin"],
                "segment_semantic_margin": row["segment_semantic_margin"],
                "global_envelope_tIoU": control["envelope_tiou"],
                "global_order_margin": control["order_margin"],
            }
        )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    import matplotlib.pyplot as plt

    adaptive_lengths = [record["argmax_N"] for record in records]
    global_lengths = [
        max(range(6), key=global_control[row["question"]]["length_prior"].__getitem__) + 1
        for row in adaptive
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    bins = np.arange(0.5, 7.0, 1)
    axes[0].hist([adaptive_lengths, global_lengths], bins=bins, label=["adaptive", "global"])
    axes[0].set_xlabel("argmax process length N")
    axes[0].set_ylabel("expressions")
    axes[0].legend()
    axes[1].scatter(
        [record["start_location"] for record in records],
        [record["end_location"] for record in records],
        s=14,
    )
    axes[1].set_xlabel("expected start")
    axes[1].set_ylabel("expected end")
    envelope_adaptive = [record["envelope_tIoU"] for record in records if record["envelope_tIoU"] is not None]
    envelope_global = [record["global_envelope_tIoU"] for record in records if record["global_envelope_tIoU"] is not None]
    axes[2].boxplot([envelope_adaptive, envelope_global], tick_labels=["adaptive", "global"])
    axes[2].set_ylabel("process envelope tIoU")
    figure.savefig(args.figure, dpi=180)

    selected = sorted(adaptive, key=lambda row: stable(row["question"]))[:12]
    figure, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    for axis, row in zip(axes.flat, selected):
        matrix = np.asarray(row["posterior_process"]).T
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1)
        axis.set_title(row["question"].split("_", 3)[-1][:42], fontsize=7)
        axis.set_xlabel("sampled frame")
        axis.set_ylabel("process state")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(args.posterior_figure, dpi=180)


if __name__ == "__main__":
    main()

"""Export final-epoch segmental diagnostics and deterministic heatmaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.training.read_text().splitlines() if line]
    final = {}
    for row in rows:
        final[row["question"]] = row
    records = []
    for row in final.values():
        posterior = row["process_posterior"]
        expected = [
            sum((state + 1) * value for state, value in enumerate(frame))
            for frame in posterior
        ]
        duration = [
            sum(frame[state] for frame in posterior)
            for state in range(len(posterior[0]))
        ]
        records.append(
            {
                "question": row["question"],
                "expression_id": row["expression_id"],
                "frames": json.dumps(row["sampled_frame_indices"]),
                "argmax_N": max(range(len(row["terminal_prior"])), key=row["terminal_prior"].__getitem__) + 1,
                "expected_N": row["expected_process_length"],
                "state_duration": json.dumps(duration),
                "posterior_entropy": row["posterior_entropy"],
                "expected_state_monotonic": all(
                    right + 1e-5 >= left for left, right in zip(expected, expected[1:])
                ),
                "largest_state_fraction": max(duration) / len(posterior),
                "order_margin": row["order_margin"],
                "object_margin": row["target_distractor_margin"],
            }
        )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    import matplotlib.pyplot as plt
    import numpy as np

    ordered = sorted(final.values(), key=lambda row: stable(row["question"]))[:6]
    figure, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for axis, row in zip(axes.flat, ordered):
        matrix = np.asarray(row["process_posterior"]).T
        image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
        axis.set_xlabel("sampled frame")
        axis.set_ylabel("process state")
        axis.set_title(row["question"].split("_", 3)[-1][:48], fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)


if __name__ == "__main__":
    main()

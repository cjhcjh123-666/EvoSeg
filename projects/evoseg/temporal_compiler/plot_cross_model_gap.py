#!/usr/bin/env python3
"""Plot model-internal Dynamic minus Static J&F with cluster-bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Long-RVOS 64-object protocol check")
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {args.summary}")

    labels = [row["model"].replace("Sa2VA-Qwen3-VL-4B", "Sa2VA-4B") for row in rows]
    values = [100.0 * float(row["dynamic_minus_static_J_and_F"]) for row in rows]
    lower = [100.0 * float(row["bootstrap_ci_low"]) for row in rows]
    upper = [100.0 * float(row["bootstrap_ci_high"]) for row in rows]
    errors = [[value - low for value, low in zip(values, lower)],
              [high - value for value, high in zip(values, upper)]]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(rows)]
    ax.bar(labels, values, color=colors, width=0.62, zorder=2)
    ax.errorbar(labels, values, yerr=errors, fmt="none", ecolor="#222222",
                elinewidth=1.5, capsize=5, zorder=3)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("Dynamic − Static J&F (percentage points)")
    ax.set_title(args.title)
    for index, value in enumerate(values):
        ax.text(index + 0.08, value + 0.12, f"{value:.2f} pp", ha="left", va="bottom",
                color="white", fontsize=9, fontweight="bold")
    ax.text(0.01, 0.01, "95% source-video cluster bootstrap CI; n=64 paired objects",
            transform=ax.transAxes, fontsize=8, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

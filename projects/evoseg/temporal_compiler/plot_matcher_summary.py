#!/usr/bin/env python3
"""Plot end-to-end J&F for the frozen-component matcher prototype."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["method"], row["description_type"]): float(row["mean_J_and_F"]) for row in rows}
    methods = ["raw_expression", "concept_only", "static", "temporal", "concept_oracle"]
    labels = ["Raw", "Concept", "Static C", "Temporal D", "Oracle"]
    x = np.arange(len(methods))
    width = 0.34
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for offset, kind, color in [(-width / 2, "static", "#4C78A8"), (width / 2, "dynamic", "#F58518")]:
        values = [100.0 * lookup[(method, kind)] for method in methods]
        bars = ax.bar(x + offset, values, width, color=color, label=kind.title())
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test J&F (%)")
    ax.set_title("Minimal frozen-component matcher prototype")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "15 held-out source videos / 40 objects; three-seed mean; candidate misses retained",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

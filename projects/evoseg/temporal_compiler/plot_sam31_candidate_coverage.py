#!/usr/bin/env python3
"""Plot complete-object SAM3.1 candidate coverage without using per-track GT data."""

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
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["description_type"] in {"static", "dynamic"}
        ]
    methods = ["raw_expression", "concept", "qwen_concept"]
    descriptions = ["static", "dynamic"]
    labels = {
        "raw_expression": "Raw expression",
        "concept": "Deterministic concept",
        "qwen_concept": "Qwen concept",
    }
    values = {
        (row["prompt_method"], row["description_type"]): 100.0
        * float(row["recall_at_0_5"])
        for row in rows
    }
    x = np.arange(len(methods))
    width = 0.34
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    for offset, description, color in [
        (-width / 2, "static", "#4C78A8"),
        (width / 2, "dynamic", "#F58518"),
    ]:
        heights = [values[(method, description)] for method in methods]
        bars = ax.bar(x + offset, heights, width, label=description.title(), color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xticks(x, [labels[method] for method in methods])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Complete-object Recall @ oracle J&F ≥ 0.5 (%)")
    ax.set_title("SAM3.1 candidate-bank coverage on Long-RVOS")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "Object-weighted; incomplete prompt/type cells excluded and audited separately",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

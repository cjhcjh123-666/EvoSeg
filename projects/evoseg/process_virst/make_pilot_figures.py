"""Create lightweight, data-derived figures for the ProcessVIRST SFT pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def method_overview(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 2.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    axis.axis("off")
    labels = [
        "Frozen VIRST\nquery + video states",
        "4 ordinal\nprocess slots",
        "Monotonic\nevent-frame DP",
        "Process-conditioned\nframe states",
        "Official SegPrompter\n+ SAM2",
    ]
    xs = [0.2, 2.65, 5.0, 7.35, 9.7]
    colors = ["#dceaf7", "#fce5cd", "#f9cb9c", "#d9ead3", "#d9d2e9"]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.8), 2.05, 1.35, boxstyle="round,pad=0.08", fc=color, ec="#444"
            )
        )
        axis.text(x + 1.025, 1.48, label, ha="center", va="center", fontsize=9)
        if index + 1 < len(xs):
            axis.annotate("", (xs[index + 1] - 0.08, 1.48), (x + 2.1, 1.48), arrowprops={"arrowstyle": "->"})
    axis.text(6.0, 2.62, "Process reasoning enters before frame-specific pixel prompts", ha="center", weight="bold")
    save(fig, output / "method_overview.png")


def result_bars(run: Path, output: Path) -> None:
    baseline = json.loads((run / "groundmore_baseline_pilot16/groundmore_summary.json").read_text())
    values = [("Official", 100 * baseline["J_and_F"])]
    for seed in (11, 23, 42):
        value = json.loads((run / f"groundmore_pilot16_seed{seed}/groundmore_summary.json").read_text())
        values.append((f"Full s{seed}", 100 * value["J_and_F"]))
    for label, directory in (
        ("No-order s11", "control_noorder_seed11"),
        ("Global s11", "control_global_seed11"),
    ):
        value = json.loads((run / f"groundmore_pilot16_{directory}/groundmore_summary.json").read_text())
        values.append((label, 100 * value["J_and_F"]))
    fig, axis = plt.subplots(figsize=(9, 4.2))
    axis.bar([x[0] for x in values], [x[1] for x in values], color=["#777"] + ["#3d85c6"] * 3 + ["#e69138", "#6aa84f"])
    axis.set_ylabel("GroundMoRe pilot J&F (%)")
    axis.set_ylim(27.6, 27.95)
    axis.tick_params(axis="x", rotation=25)
    for index, (_, value) in enumerate(values):
        axis.text(index, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_title("No stable segmentation gain from ordered SFT")
    save(fig, output / "groundmore_results.png")


def order_figures(run: Path, output: Path) -> None:
    conditions = []
    for label, directory in (
        ("Full s11", "groundmore_pilot16_seed11"),
        ("Full s23", "groundmore_pilot16_seed23"),
        ("Full s42", "groundmore_pilot16_seed42"),
        ("No-order", "groundmore_pilot16_control_noorder_seed11"),
        ("Global", "groundmore_pilot16_control_global_seed11"),
    ):
        conditions.append((label, json.loads((run / directory / "order_summary.json").read_text())))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(conditions))
    for offset, permutation, color in ((-0.16, "reverse", "#3d85c6"), (0.16, "block_swap", "#e69138")):
        means = [item[1][permutation]["mean"] for item in conditions]
        lows = [mean - item[1][permutation]["ci_low"] for mean, item in zip(means, conditions)]
        highs = [item[1][permutation]["ci_high"] - mean for mean, item in zip(means, conditions)]
        axis.errorbar(x + offset, means, yerr=[lows, highs], fmt="o", capsize=4, color=color, label=permutation)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(x, [item[0] for item in conditions], rotation=20)
    axis.set_ylabel("Original − permuted alignment score")
    axis.set_title("Order preference is not stable across seeds")
    axis.legend()
    save(fig, output / "order_margin.png")

    rows = load_csv(run / "groundmore_pilot16_seed11/order_diagnostics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, permutation in zip(axes, ("reverse", "block_swap")):
        subset = [row for row in rows if row["permutation"] == permutation]
        original = np.array([float(row["original_alignment_score"]) for row in subset])
        permuted = np.array([float(row["permuted_alignment_score"]) for row in subset])
        axis.scatter(permuted, original, s=18, alpha=0.7)
        low, high = min(original.min(), permuted.min()), max(original.max(), permuted.max())
        axis.plot([low, high], [low, high], "k--", lw=1)
        axis.set_xlabel(f"{permutation} score")
        axis.set_ylabel("original score")
        axis.set_title(f"seed11: {permutation}")
    save(fig, output / "normal_vs_permuted.png")


def long_baseline(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4))
    labels = ["Static", "Dynamic", "Hybrid"]
    values = [59.37, 59.00, 59.01]
    axis.bar(labels, values, color=["#6aa84f", "#3d85c6", "#8e7cc3"])
    axis.set_ylim(58.5, 59.7)
    axis.set_ylabel("Official VIRST J&F (%)")
    axis.set_title("Long-RVOS prerequisite; ProcessVIRST N/A after early stop")
    for index, value in enumerate(values):
        axis.text(index, value + 0.03, f"{value:.2f}", ha="center")
    save(fig, output / "dynamic_results.png")


def qualitative_alignment(run: Path, output: Path) -> None:
    mapping = json.loads((run / "groundmore_pilot16_dataset/groundmore_mapping.json").read_text())
    diagnostics = [json.loads(line) for line in (run / "groundmore_pilot16_seed11/process_diagnostics.jsonl").read_text().splitlines()]
    selected = [
        ("9KhJO4UloIE_0404_0414", "7"),
        ("pFX4CC5UiOY_0915_0925", "9"),
        ("NomMansxnQM_0252_0302", "3"),
    ]
    lookup = {(row["video_id"], row["expression_id"]): index for index, row in enumerate(mapping)}
    for case_index, key in enumerate(selected, start=1):
        index = lookup[key]
        value = diagnostics[index]
        arrays = [
            np.array(value["alignment"])[0],
            np.array(value["permutations"]["reverse"]["alignment"])[0],
            np.array(value["permutations"]["block_swap"]["alignment"])[0],
        ]
        fig, axes = plt.subplots(3, 1, figsize=(10, 5.8), sharex=True)
        for axis, array, title in zip(axes, arrays, ("Original", "Reverse", "BlockSwap")):
            image = axis.imshow(array, aspect="auto", cmap="viridis", vmin=0, vmax=max(a.max() for a in arrays))
            axis.set_ylabel("slot")
            axis.set_title(title, loc="left", fontsize=9)
        axes[-1].set_xlabel("sampled frame index")
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
        fig.suptitle(mapping[index]["question"], fontsize=9)
        fig.savefig(output / "qualitative_order_cases" / f"case_{case_index}_alignment.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qualitative_order_cases").mkdir(exist_ok=True)
    method_overview(args.output_dir)
    result_bars(args.run_root, args.output_dir)
    order_figures(args.run_root, args.output_dir)
    long_baseline(args.output_dir)
    qualitative_alignment(args.run_root, args.output_dir)


if __name__ == "__main__":
    main()

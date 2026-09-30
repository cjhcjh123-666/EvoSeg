"""Publish lightweight reports, figures, and qualitative cases for FTSG."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


LABELS = {
    "sam31_raw_expression": "SAM3.1 raw expression",
    "sam31_concept_native": "SAM3.1 concept native",
    "static_identity": "Static identity",
    "temporal_identity": "Temporal identity (FTSG)",
    "oracle_identity": "ORACLE candidate identity",
    "temporal_order_shuffle": "Order-shuffled temporal",
    "temporal_mean_pool": "Mean-pool temporal",
}


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def pct(value) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}"


def summary_index(summary: dict) -> dict[tuple[str, str], dict]:
    return {(row["description_type"], row["method"]): row for row in summary["method_summary"]}


def ci_line(value: dict) -> str:
    return f"{100*value['mean']:+.2f} pp, 95% CI [{100*value['ci_low']:+.2f}, {100*value['ci_high']:+.2f}]"


def table(rows: list[dict]) -> str:
    lines = ["| Type | Method | Evaluated | Sel. acc. on hits | J | F | J&F |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['description_type']} | {LABELS[row['method']]} | {row['evaluated_expressions']}/{row['expressions']} | {pct(row['selection_accuracy_on_hits'])} | {pct(row['J'])} | {pct(row['F'])} | {pct(row['J_and_F'])} |"
        )
    return "\n".join(lines)


def decode(rle: dict) -> np.ndarray:
    value = dict(rle)
    value["counts"] = value["counts"].encode("ascii")
    return mask_utils.decode(value).astype(bool)


def candidate_mask(path: str | None, track_id: str | None, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    if not path or track_id in (None, "", "None") or not Path(path).is_file():
        return np.zeros(shape, dtype=bool)
    payload = json.loads(Path(path).read_text())
    try:
        position = payload["evaluation_frame_indices"].index(frame_index)
    except ValueError:
        return np.zeros(shape, dtype=bool)
    for track in payload["tracks"]:
        if int(track["track_id"]) == int(track_id):
            return decode(track["frames"][position])
    return np.zeros(shape, dtype=bool)


def make_method_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    boxes = [
        (.04, .58, .17, .23, "Video + query"),
        (.29, .69, .22, .20, "Temporal identity branch\nwhich object?"),
        (.29, .23, .22, .20, "Frozen SAM3.1\ncandidate tracks"),
        (.60, .46, .18, .20, "Track scores\nargmax identity"),
        (.84, .46, .13, .20, "Direct mask\ntrack output"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#e8f1fb", edgecolor="#24527a", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [((.21, .69), (.29, .79)), ((.21, .69), (.29, .33)), ((.51, .79), (.60, .58)), ((.51, .33), (.60, .54)), ((.78, .56), (.84, .56))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    ax.text(.5, .05, "Temporal for identity; spatial tracks for pixels", ha="center", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_case_figures(docs: Path, run_config: dict, selection: list[dict]) -> dict[str, int]:
    manifest = json.loads(Path(run_config["inputs"]["manifest"]).read_text())
    feature_meta = json.loads(Path(run_config["inputs"]["eval_features"]).with_suffix(".json").read_text())
    candidates = {row["identity"]: row.get("candidate_tracks_path") for row in feature_meta["records"]}
    objects = {f"{item['dataset']}/{item['video_id']}/{item['object_id']}": item for item in manifest["objects"]}
    groups = {
        "temporal_correction": [row for row in selection if row["concept_candidate_hit"] == "1" and row["static_identity_selection_correct"] == "0" and row["temporal_identity_selection_correct"] == "1"],
        "both_correct": [row for row in selection if row["static_identity_selection_correct"] == "1" and row["temporal_identity_selection_correct"] == "1"],
        "temporal_damage": [row for row in selection if row["static_identity_selection_correct"] == "1" and row["temporal_identity_selection_correct"] == "0"],
        "candidate_miss": [row for row in selection if row["concept_candidate_hit"] == "0"],
    }
    out = docs / "figures/qualitative_cases"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    counts = {}
    for label, values in groups.items():
        chosen = sorted(values, key=lambda row: row["identity"])[:3]
        counts[label] = len(chosen)
        for row in chosen:
            item = objects["/".join(row["identity"].split("/")[:3])]
            positions = [index for index, value in enumerate(item["evaluation_mask_paths"]) if Path(value).is_file()]
            if not positions:
                continue
            position = positions[-1]
            frame_index = item["evaluation_frame_indices"][position]
            frame_name = item["frame_names"][frame_index]
            image = np.asarray(Image.open(Path(manifest["dataset"]["image_root"]) / item["video_id"] / f"{frame_name}.jpg").convert("RGB"))
            gt = np.asarray(Image.open(item["evaluation_mask_paths"][position]).convert("L")) > 0
            static = candidate_mask(candidates.get(row["identity"]), row["static_identity_selected_track_id"], frame_index, gt.shape)
            temporal = candidate_mask(candidates.get(row["identity"]), row["temporal_identity_selected_track_id"], frame_index, gt.shape)
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
            for ax, value, title in zip(axes, (image, gt, static, temporal), ("Frame", "GT", "Static selected", "Temporal selected")):
                ax.imshow(value, cmap=None if value.ndim == 3 else "gray"); ax.set_title(title); ax.axis("off")
            fig.suptitle(f"{label}: {row['expression']}", fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"{label}__{row['identity'].replace('/','__')}.png", dpi=140); plt.close(fig)
    return counts


def run(args) -> int:
    root = Path(args.run_dir)
    docs = Path(args.docs)
    figures = docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    results = read_csv(root / "per_expression_results.csv")
    selection = read_csv(root / "candidate_selection.csv")
    index = summary_index(summary)
    for name in ("per_expression_results.csv", "candidate_selection.csv", "temporal_controls.csv", "efficiency.csv"):
        shutil.copy2(root / name, docs / name)

    method = """# FTSG method

Factorized Temporal-Spatial Grounding explicitly separates **which object** from **which pixels**. A frozen Sa2VA cumulative state and ordered candidate-region sequence feed the verified single-layer BiGRU identity selector. The selected frozen SAM3.1 concept candidate track is returned directly; temporal state never generates a heatmap or pixel prompt. Sa2VA, SAM3.1, and visual encoders remain frozen. GT is used only on official train to assign a candidate identity label and only at evaluation to score outputs.

Controls use the identical candidate bank and direct-track executor. Static identity sees only the final anchor appearance; order shuffle permutes only the four candidate-region features with a fixed per-expression derangement; mean pool removes order using a parameter-matched scorer.
"""
    (docs / "METHOD.md").write_text(method)
    mean_training = summary["mean_pool_training"]
    (docs / "TRAINING.md").write_text(
        "# Training\n\n"
        "The preregistered Static/Temporal BiGRU checkpoints for seeds 11/23/42 were loaded without validation retraining. The official-train split, LR 3e-4, 40-epoch cap, patience 6, hidden sizes, and hit threshold 0.3 are unchanged.\n\n"
        + "\n".join(f"- Mean-pool seed {row['seed']}: complete, best epoch {row['best_epoch']}, {row['parameter_count']:,} parameters, {row['elapsed_seconds']:.1f}s." for row in mean_training)
        + "\n"
    )
    main = summary["comparisons"]["dynamic"]
    full = "# Full validation results\n\n" + table([row for row in summary["method_summary"] if row["method"] in {"sam31_raw_expression", "sam31_concept_native", "static_identity", "temporal_identity", "oracle_identity"}])
    full += f"\n\n## Dynamic primary comparison\n\n- Selection accuracy: {ci_line(main['temporal_minus_static_accuracy'])}.\n- Candidate-direct J&F: {ci_line(main['temporal_minus_static_J_and_F'])}.\n- Coverage: {summary['coverage']['objects']} paired objects, {summary['coverage']['expressions']} official expressions; {summary['coverage']['state_failures']} state failure retained as a zero-output paired failure.\n"
    (docs / "FULL_RESULTS.md").write_text(full)
    controls = f"""# Temporal controls

On Dynamic expressions:

- Normal temporal − fixed order shuffle selection accuracy: {ci_line(main['temporal_minus_order_shuffle_accuracy'])}.
- Normal temporal − fixed order shuffle J&F: {ci_line(main['temporal_minus_order_shuffle_J_and_F'])}.
- Normal temporal − parameter-matched mean pool selection accuracy: {ci_line(main['temporal_minus_mean_pool_accuracy'])}.
- Normal temporal − parameter-matched mean pool J&F: {ci_line(main['temporal_minus_mean_pool_J_and_F'])}.

Order shuffle keeps the candidate, frame set, spatial content, query, and cumulative Sa2VA states fixed; only the four candidate-region features are deterministically permuted. Mean pooling is trained on the same official-train split with seeds 11/23/42 and differs from the BiGRU only in its order-free aggregation/scorer.
"""
    (docs / "TEMPORAL_CONTROLS.md").write_text(controls)
    efficiency = read_csv(root / "efficiency.csv")
    efficiency_lines = ["# Efficiency", "", "| Component | Params/head | Ensemble | Total selector params | Latency/query (s) | Peak memory (GiB) |", "|---|---:|---:|---:|---:|---:|"]
    for row in efficiency:
        peak = "N/A" if not row["peak_memory_bytes"] else f"{float(row['peak_memory_bytes'])/2**30:.2f}"
        latency = "N/A" if not row["latency_seconds_per_query"] else f"{float(row['latency_seconds_per_query']):.4f}"
        efficiency_lines.append(f"| {row['component']} | {row['params_per_head']} | {row['ensemble_size']} | {row['params_total_loaded']} | {latency} | {peak} |")
    efficiency_lines += ["", "Candidate generation is reported separately and is not hidden inside selector-only overhead."]
    (docs / "EFFICIENCY.md").write_text("\n".join(efficiency_lines) + "\n")
    misses = config["candidate_generation_failures"]
    native_missing = {method: sum(row["method"] == method and row["status"] == "native_score_unavailable" for row in results) for method in ("sam31_raw_expression", "sam31_concept_native")}
    (docs / "FAILURE_ANALYSIS.md").write_text(
        f"# Failure analysis\n\n- Missing temporal-state expressions: {len(config['missing_state_identities'])}: " + ", ".join(f"`{x}`" for x in config["missing_state_identities"]) + ".\n"
        f"- Candidate-generation failures: {len(misses)}: " + ", ".join(f"`{x}`" for x in misses) + ".\n"
        "- Candidate misses are retained with zero end-to-end output; no false positive label is manufactured.\n"
        "- The missing-state expression is retained as a paired zero output for identity selectors, so it cannot improve the reported delta by deletion.\n"
        f"- Official native confidence is unavailable for {native_missing['sam31_raw_expression']} raw-expression and {native_missing['sam31_concept_native']} concept records; those baseline cells are reported N/A and excluded from that baseline's mean.\n"
    )
    rationale = {
        "GO": "Both primary Dynamic confidence intervals are positive, and at least one ordered control has a positive confidence interval.",
        "PARTIAL GO": "The primary Dynamic selector gain is stable, but the ordered controls do not establish temporal-order reasoning. The supported claim is multi-frame identity grounding.",
        "NO-GO": "The full-validation primary Dynamic advantage is not stable; no capacity or hyperparameter rescue was attempted.",
    }[summary["decision"]]
    refiner = "Not started; identity result did not meet full GO." if summary["decision"] != "GO" else "Not started automatically: the frozen direct-track executor is the preregistered primary method; spatial refinement requires separate review."
    (docs / "FINAL_GO_NOGO.md").write_text(f"# Final decision\n\n**{summary['decision']}**\n\n{rationale}\n\nSpatial Refiner: {refiner}\n")

    make_method_figure(figures / "method_overview_draft.png")
    dynamic_methods = ["static_identity", "temporal_identity"]
    fig, ax = plt.subplots(figsize=(5, 3.5)); values = [index[("dynamic", method)]["J_and_F"] * 100 for method in dynamic_methods]
    ax.bar(["Static", "Temporal"], values, color=["#999", "#2a6fbb"]); ax.set_ylabel("Dynamic J&F"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figures / "dynamic_main_result.png", dpi=170); plt.close(fig)
    headroom_methods = ["sam31_concept_native", "static_identity", "temporal_identity", "oracle_identity"]
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.bar(["Native", "Static", "Temporal", "Oracle"], [index[("dynamic", method)]["J_and_F"] * 100 for method in headroom_methods], color=["#aaa", "#777", "#2a6fbb", "#e39c2c"]); ax.set_ylabel("Dynamic candidate-direct J&F"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figures / "oracle_headroom.png", dpi=170); plt.close(fig)
    control_methods = ["temporal_identity", "temporal_order_shuffle", "temporal_mean_pool"]
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.bar(["Ordered BiGRU", "Order shuffle", "Mean pool"], [index[("dynamic", method)]["J_and_F"] * 100 for method in control_methods], color=["#2a6fbb", "#c45a4a", "#58a06b"]); ax.set_ylabel("Dynamic J&F"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figures / "temporal_control.png", dpi=170); plt.close(fig)
    case_counts = make_case_figures(docs, config, selection)
    (docs / "qualitative_cases.json").write_text(json.dumps(case_counts, indent=2) + "\n")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--docs", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

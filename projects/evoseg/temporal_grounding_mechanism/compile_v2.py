"""Compile full-probe and pixel-execution diagnostics into review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.temporal_grounding_mechanism.common import (
    object_weighted,
    source_video_bootstrap,
    write_csv,
)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def latest_success(paths: list[Path]) -> list[dict]:
    rows = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                value = json.loads(line)
                if value.get("status") == "success":
                    rows[(value["identity"], value["identity_basis"], value["prompt_form"])] = value
    return list(rows.values())


def failed_attempt_count(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        if not path.is_file():
            continue
        with path.open() as handle:
            count += sum(json.loads(line).get("status") == "failed" for line in handle)
    return count


def mean_objects(rows: list[dict], fields: list[str]) -> dict[str, float | None]:
    values = object_weighted(rows, fields)
    return {
        field: float(np.mean([float(row[field]) for row in values])) if values else None
        for field in fields
    }


def probe_summary(rows: list[dict]) -> list[dict]:
    output = []
    for seed in ("ensemble", "11", "23", "42"):
        seed_rows = [row for row in rows if str(row["model_seed"]) == seed]
        for kind in ("static", "dynamic", "hybrid"):
            selected = [row for row in seed_rows if row["description_type"] == kind]
            hits = [row for row in selected if int(row["candidate_hit"])]
            for method in ("static", "temporal"):
                accuracy = mean_objects(hits, [f"{method}_selection_correct"])[f"{method}_selection_correct"]
                jf = mean_objects(selected, [f"{method}_candidate_direct_J_and_F"])[f"{method}_candidate_direct_J_and_F"]
                output.append({
                    "model_seed": seed,
                    "description_type": kind,
                    "method": method,
                    "expressions": len(selected),
                    "candidate_hit_expressions": len(hits),
                    "candidate_miss_expressions": len(selected) - len(hits),
                    "objects": len({(row["video_id"], row["object_id"]) for row in selected}),
                    "videos": len({row["video_id"] for row in selected}),
                    "selection_accuracy_on_candidate_hits": accuracy,
                    "candidate_direct_J_and_F": jf,
                })
    return output


def pixel_summary(rows: list[dict]) -> list[dict]:
    fields = [
        "anchor_candidate_J", "anchor_candidate_F", "anchor_candidate_J_and_F",
        "init_J", "init_F", "init_J_and_F", "candidate_to_init_IoU",
        "propagated_J", "propagated_F", "propagated_J_and_F",
        "initialization_loss_J_and_F", "propagation_loss_J_and_F",
        "latency_seconds_synchronized", "peak_memory_bytes",
    ]
    output = []
    for basis in ("predicted_static", "oracle_identity"):
        for form in ("point", "box"):
            for kind in ("static", "dynamic", "hybrid"):
                selected = [
                    row for row in rows
                    if row["identity_basis"] == basis and row["prompt_form"] == form
                    and row["description_type"] == kind
                ]
                output.append({
                    "identity_basis": basis, "prompt_form": form, "description_type": kind,
                    "expressions": len(selected),
                    "objects": len({(row["video_id"], row["object_id"]) for row in selected}),
                    **mean_objects(selected, fields),
                })
    return output


def decomposition_summary(rows: list[dict]) -> list[dict]:
    fields = [
        "direct_track_J_and_F", "anchor_candidate_J_and_F", "init_J_and_F",
        "candidate_to_init_IoU", "propagated_J_and_F",
        "initialization_loss_J_and_F", "propagation_loss_J_and_F",
        "direct_to_propagated_loss_J_and_F",
    ]
    output = []
    for basis in ("predicted_static", "oracle_identity"):
        for form in ("point", "box"):
            for kind in ("static", "dynamic", "hybrid"):
                selected = [
                    row for row in rows
                    if row["identity_basis"] == basis and row["prompt_form"] == form
                    and row["description_type"] == kind
                ]
                output.append({
                    "identity_basis": basis,
                    "prompt_form": form,
                    "description_type": kind,
                    "expressions": len(selected),
                    "objects": len({(row["video_id"], row["object_id"]) for row in selected}),
                    **mean_objects(selected, fields),
                })
    return output


def markdown_table(rows: list[dict], fields: list[str], percent: set[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value * 100:.2f}" if field in percent else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args) -> None:
    probe_dir = Path(args.probe_dir)
    pixel_root = Path(args.pixel_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detailed_probe = read_csv(probe_dir / "probe_results_by_model.csv")
    ensemble = [row for row in detailed_probe if row["model_seed"] == "ensemble"]
    probe_audit = json.loads((probe_dir / "probe_audit.json").read_text())
    expected_evaluation = int(probe_audit["evaluation_expressions"])
    if len(ensemble) != expected_evaluation:
        raise RuntimeError(f"full probe incomplete: ensemble {len(ensemble)}/{expected_evaluation}")
    if len(detailed_probe) != expected_evaluation * 4:
        raise RuntimeError(
            f"full probe seed matrix incomplete: {len(detailed_probe)}/{expected_evaluation * 4}"
        )
    if probe_audit["checkpoint_mode"] != "load_only":
        raise RuntimeError("full validation was not evaluated from frozen reconstructed checkpoints")
    eval_feature_meta = json.loads(
        Path(probe_audit["inputs"]["eval_features"]).with_suffix(".json").read_text()
    )
    missing_states = eval_feature_meta.get("missing_state_identities", [])
    missing_candidates = eval_feature_meta.get("missing_candidate_identities", [])
    shutil.copyfile(probe_dir / "probe_results_by_model.csv", output / "full_probe_results.csv")
    probe_stats = probe_summary(detailed_probe)

    pixel_paths = sorted(pixel_root.glob("pixel*_shard_*/pixel_execution_results.jsonl"))
    pixel = latest_success(pixel_paths)
    pixel_failed_attempts = failed_attempt_count(pixel_paths)
    expected = 275 * 2 * 2
    if len(pixel) != expected:
        raise RuntimeError(f"pixel condition matrix incomplete: {len(pixel)}/{expected}")
    init_rows = []
    for row in pixel:
        for stage in row["stage_metrics"]:
            init_rows.append({
                key: row[key]
                for key in ("identity", "dataset", "video_id", "object_id", "expression_id", "description_type", "identity_basis", "prompt_form", "ORACLE")
            } | stage)
    write_csv(output / "prompt_initialization.csv", init_rows)

    offline = read_csv(Path(args.offline_ladder))
    direct = {}
    for row in offline:
        if row["condition"] == "PREDICTED_CANDIDATE_DIRECT_STATIC":
            direct[(row["identity"], "predicted_static")] = row
        elif row["condition"] == "ORACLE_CANDIDATE":
            direct[(row["identity"], "oracle_identity")] = row
    decomposition = []
    for row in pixel:
        source = direct[(row["identity"], row["identity_basis"])]
        decomposition.append({
            key: row[key]
            for key in ("identity", "dataset", "video_id", "object_id", "expression_id", "description_type", "identity_basis", "prompt_form", "ORACLE")
        } | {
            "direct_track_J": source["J"], "direct_track_F": source["F"], "direct_track_J_and_F": source["J_and_F"],
            "anchor_candidate_J": row["anchor_candidate_J"], "anchor_candidate_F": row["anchor_candidate_F"],
            "anchor_candidate_J_and_F": row["anchor_candidate_J_and_F"],
            "init_J": row["init_J"], "init_F": row["init_F"], "init_J_and_F": row["init_J_and_F"],
            "candidate_to_init_IoU": row["candidate_to_init_IoU"],
            "propagated_J": row["propagated_J"], "propagated_F": row["propagated_F"], "propagated_J_and_F": row["propagated_J_and_F"],
            "initialization_loss_J_and_F": row["initialization_loss_J_and_F"],
            "propagation_loss_J_and_F": row["propagation_loss_J_and_F"],
            "direct_to_propagated_loss_J_and_F": float(row["propagated_J_and_F"]) - float(source["J_and_F"]),
            "latency_seconds_synchronized": row["latency_seconds_synchronized"],
            "peak_memory_bytes": row["peak_memory_bytes"], "cache_hit": row["cache_hit"],
            "prior_point_propagated_J_and_F": row.get("prior_point_propagated_J_and_F"),
            "rerun_minus_prior_J_and_F": row.get("rerun_minus_prior_J_and_F"),
        })
    write_csv(output / "propagation_decomposition.csv", decomposition)
    pixel_stats = pixel_summary(pixel)
    decomposition_stats = decomposition_summary(decomposition)

    dynamic_ensemble = [row for row in ensemble if row["description_type"] == "dynamic"]
    dynamic_hits = [row for row in dynamic_ensemble if int(row["candidate_hit"])]
    accuracy_ci = source_video_bootstrap(dynamic_hits, "temporal_selection_correct", "static_selection_correct")
    jf_ci = source_video_bootstrap(dynamic_ensemble, "temporal_candidate_direct_J_and_F", "static_candidate_direct_J_and_F")
    full_report = f"""# Full Temporal Probe Validation

The preregistered, parameter-identical probes were evaluated on all 274 paired validation objects without validation training, seed selection, architecture changes, or hyperparameter changes. {expected_evaluation}/1189 expressions produced evaluable frozen inputs; {len(missing_states)} deterministic state-extraction failure(s) are listed below and were not imputed. The {len(missing_candidates)} official candidate-generation failure(s) were retained as genuine no-candidate outcomes (zero candidate-direct J&F), rather than silently removed. The original implementation did not persist pilot checkpoints; the three probes were therefore deterministically reconstructed once from the unchanged official-train tensors/split/config, checked against the pilot outputs, frozen, and loaded in `load_only` mode for this full evaluation.

""" + markdown_table(
        [row for row in probe_stats if row["model_seed"] == "ensemble"],
        ["description_type", "method", "expressions", "candidate_hit_expressions", "candidate_miss_expressions", "selection_accuracy_on_candidate_hits", "candidate_direct_J_and_F"],
        {"selection_accuracy_on_candidate_hits", "candidate_direct_J_and_F"},
    ) + f"""

## Dynamic paired effect

- Selection accuracy Temporal − Static: **{accuracy_ci['mean'] * 100:+.2f} pp**, source-video bootstrap 95% CI **[{accuracy_ci['ci_low'] * 100:+.2f}, {accuracy_ci['ci_high'] * 100:+.2f}] pp** ({accuracy_ci['objects']} hit objects, {accuracy_ci['videos']} videos; 2,000 resamples).
- Candidate-direct J&F Temporal − Static: **{jf_ci['mean'] * 100:+.2f} pp**, source-video bootstrap 95% CI **[{jf_ci['ci_low'] * 100:+.2f}, {jf_ci['ci_high'] * 100:+.2f}] pp** ({jf_ci['objects']} objects, {jf_ci['videos']} videos; 2,000 resamples).

## Preregistered seeds

""" + markdown_table(
        [row for row in probe_stats if row["model_seed"] != "ensemble" and row["description_type"] == "dynamic"],
        ["model_seed", "method", "selection_accuracy_on_candidate_hits", "candidate_direct_J_and_F"],
        {"selection_accuracy_on_candidate_hits", "candidate_direct_J_and_F"},
    ) + "\n\n## State extraction failures\n\n" + (
        "\n".join(f"- `{identity}`" for identity in missing_states) if missing_states else "None."
    ) + "\n\n## Candidate-generation failures (counted as misses)\n\n" + (
        "\n".join(f"- `{identity}`" for identity in missing_candidates)
        if missing_candidates else "None."
    ) + "\n"
    (output / "FULL_PROBE_REPORT.md").write_text(full_report)

    pixel_report = f"""# Pixel Execution Decomposition

The candidate identity and four-stage plan are fixed across C0/C1/C2 and C3/C4. Candidate masks generate prompts; GT is opened only after prompt inference. The matrix contains all **{len(pixel)}/1100 unique successful conditions**. It preserves **{pixel_failed_attempts} failed attempt(s)** separately; their keys were retried successfully and are counted once in the matrix. Point and box use the Meta public multiplex `add_prompt` wrapper. The public multiplex model has no stable mask-prompt method (`add_mask` explicitly rejects it), so MASK_INIT / MASK_PROPAGATE are **N/A**. Box prompting uses official semantic-box behavior, which resets semantic state at each stage; this limitation is part of the measured public interface.

""" + markdown_table(
        decomposition_stats,
        ["identity_basis", "prompt_form", "description_type", "expressions",
         "direct_track_J_and_F", "anchor_candidate_J_and_F", "init_J_and_F",
         "candidate_to_init_IoU", "propagated_J_and_F",
         "initialization_loss_J_and_F", "propagation_loss_J_and_F",
         "direct_to_propagated_loss_J_and_F"],
        {"direct_track_J_and_F", "anchor_candidate_J_and_F", "init_J_and_F",
         "candidate_to_init_IoU", "propagated_J_and_F",
         "initialization_loss_J_and_F", "propagation_loss_J_and_F",
         "direct_to_propagated_loss_J_and_F"},
    ) + """

`initialization_loss_J_and_F = init - anchor`; `propagation_loss_J_and_F = propagated - init`; and `direct_to_propagated_loss_J_and_F = propagated - C0 DIRECT_TRACK`. Negative values are losses. C1/C3 score only the immediate anchor output, whereas C2/C4 score the propagated video.
"""
    (output / "PIXEL_EXECUTION_REPORT.md").write_text(pixel_report)

    decision_names = {
        "A": "PROMPT COMPRESSION BOTTLENECK",
        "B": "PROPAGATION BOTTLENECK",
        "C": "BOTH",
        "D": "NEITHER",
    }
    probe_support = jf_ci["ci_low"] > 0 and accuracy_ci["ci_low"] > 0
    final = f"""# Final Interface Diagnosis

## Decision: {args.decision} — {decision_names[args.decision]}

Full-validation Dynamic Temporal − Static is {jf_ci['mean'] * 100:+.2f} pp J&F (95% CI [{jf_ci['ci_low'] * 100:+.2f}, {jf_ci['ci_high'] * 100:+.2f}]) and {accuracy_ci['mean'] * 100:+.2f} pp selection accuracy (95% CI [{accuracy_ci['ci_low'] * 100:+.2f}, {accuracy_ci['ci_high'] * 100:+.2f}]). The statement that temporal information is present/readable is therefore {'stably supported on full validation because both source-video confidence intervals are positive' if probe_support else 'not stably supported on full validation'}.

The pixel diagnosis uses the measured point/box initialization and propagation deltas in `PIXEL_EXECUTION_REPORT.md`; mask prompting is N/A for the official multiplex public API. {args.decision_rationale}

## Next Method

{args.next_method}

No final Method was trained in this run.
"""
    (output / "FINAL_INTERFACE_DIAGNOSIS.md").write_text(final)
    (output / "summary.json").write_text(json.dumps({
        "probe_summary": probe_stats, "dynamic_accuracy_bootstrap": accuracy_ci,
        "dynamic_candidate_direct_bootstrap": jf_ci, "pixel_summary": pixel_stats,
        "decomposition_summary": decomposition_stats,
        "missing_candidate_identities": missing_candidates,
        "missing_state_identities": missing_states,
        "pixel_unique_successes": len(pixel),
        "pixel_failed_attempts_preserved": pixel_failed_attempts,
        "decision": args.decision,
    }, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--pixel-root", required=True)
    parser.add_argument("--offline-ladder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision", required=True, choices="ABCD")
    parser.add_argument("--decision-rationale", required=True)
    parser.add_argument("--next-method", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

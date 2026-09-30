"""Run frozen FTSG selectors, controls, baselines, and full validation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.evoseg.factorized_temporal_spatial.protocol import (
    count_parameters,
    fixed_derangement,
    load_mean_pool,
    predict,
    save_mean_pool,
    train_mean_pool,
)
from projects.evoseg.temporal_grounding_mechanism.common import (
    object_weighted,
    source_video_bootstrap,
    write_csv,
)
from projects.evoseg.temporal_grounding_mechanism.train_probe import (
    ProbeFactory,
    examples,
    load_models,
    predict as predict_probe,
)


METHODS = (
    "sam31_raw_expression",
    "sam31_concept_native",
    "static_identity",
    "temporal_identity",
    "oracle_identity",
    "temporal_order_shuffle",
    "temporal_mean_pool",
)


def status(root: Path, state: str, phase: str, **extra) -> None:
    value = {
        "state": state,
        "phase": phase,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    temporary = root / "STATUS.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(root / "STATUS.json")
    (root / "PROGRESS.md").write_text(
        f"# FTSG progress\n\n- Updated: {value['updated_at']}\n- Phase: `{phase}`\n- State: `{state}`\n"
    )


def read_metrics(path: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            identity = "/".join((row["dataset"], row["video_id"], row["object_id"], row["expression_id"]))
            row["candidate_metrics"] = json.loads(row["candidate_metrics"])
            result[identity][row["prompt_method"]] = row
    return result


def read_candidate_records(root: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted(root.glob("shard-*/candidate_records.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success":
                    identity = "/".join((row["dataset"], row["video_id"], str(row["object_id"]), str(row["expression_id"])))
                    result[(identity, row["prompt_method"])] = row
    return result


def candidate_lookup(metric: dict | None) -> dict[int, dict]:
    return {int(row["track_id"]): row for row in metric["candidate_metrics"]} if metric else {}


def native_top(metric: dict | None) -> int | None:
    values = candidate_lookup(metric)
    if not values:
        return None
    scored = {
        track_id: float(row["confidence"])
        for track_id, row in values.items()
        if row.get("confidence") is not None and np.isfinite(float(row["confidence"]))
    }
    return max(scored, key=scored.get) if scored else None


def oracle_id(metric: dict | None, threshold: float) -> int | None:
    if not metric or not metric.get("oracle_track_id") or float(metric["oracle_J_and_F"]) < threshold:
        return None
    return int(metric["oracle_track_id"])


def selection_row(meta: dict, method: str, metric: dict | None, selected: int | None, threshold: float, status_value: str = "success") -> dict:
    candidates = candidate_lookup(metric)
    oracle = oracle_id(metric, threshold)
    selected_metric = candidates.get(selected, {})
    return {
        "identity": meta["identity"],
        "dataset": meta["dataset"],
        "split": meta["split"],
        "video_id": meta["video_id"],
        "object_id": meta["object_id"],
        "expression_id": meta["expression_id"],
        "description_type": meta["description_type"],
        "expression": meta["expression"],
        "method": method,
        "status": status_value,
        "candidate_count": len(candidates),
        "candidate_hit": int(oracle is not None),
        "oracle_track_id": oracle,
        "selected_track_id": selected,
        "selection_correct": int(oracle is not None and selected == oracle),
        "J": float(selected_metric.get("J", 0.0)),
        "F": float(selected_metric.get("F", 0.0)),
        "J_and_F": float(selected_metric.get("J_and_F", 0.0)),
    }


def selected_from_logits(example: dict, logits: np.ndarray | None) -> int | None:
    ids = [int(value) for value in example.get("candidate_track_ids", [])]
    return ids[int(np.argmax(logits))] if logits is not None and len(logits) else None


def split_train(train_hits: list[dict]) -> tuple[list[dict], list[dict]]:
    videos = sorted({row["video_id"] for row in train_hits})
    random.Random(42).shuffle(videos)
    validation_videos = set(videos[: max(8, len(videos) // 5)])
    return (
        [row for row in train_hits if row["video_id"] not in validation_videos],
        [row for row in train_hits if row["video_id"] in validation_videos],
    )


def paired_bootstrap(rows: list[dict], left: str, right: str, metric: str, kind: str, hits_only: bool = False) -> dict:
    by_identity = defaultdict(dict)
    metadata = {}
    for row in rows:
        if row["description_type"] != kind or row["method"] not in {left, right}:
            continue
        by_identity[row["identity"]][row["method"]] = float(row[metric])
        metadata[row["identity"]] = row
    paired = []
    for identity, values in by_identity.items():
        row = metadata[identity]
        if len(values) != 2 or (hits_only and not int(row["candidate_hit"])):
            continue
        paired.append({
            "video_id": row["video_id"], "object_id": row["object_id"],
            "description_type": row["description_type"], left: values[left], right: values[right],
        })
    return source_video_bootstrap(paired, left, right)


def method_summary(rows: list[dict]) -> list[dict]:
    output = []
    for kind in ("static", "dynamic", "hybrid"):
        for method in METHODS:
            retained = [row for row in rows if row["description_type"] == kind and row["method"] == method]
            if not retained:
                continue
            unavailable = sum(row["status"] == "native_score_unavailable" for row in retained)
            selected = [row for row in retained if row["status"] != "native_score_unavailable"]
            objects = object_weighted(selected, ["J", "F", "J_and_F"])
            hits = [row for row in selected if int(row["candidate_hit"])]
            hit_objects = object_weighted(hits, ["selection_correct"]) if hits else []
            output.append({
                "description_type": kind,
                "method": method,
                "expressions": len(retained),
                "evaluated_expressions": len(selected),
                "native_score_unavailable": unavailable,
                "objects": len(objects),
                "candidate_hit_expressions": len(hits),
                "candidate_miss_expressions": len(selected) - len(hits),
                "candidate_recall": float(np.mean([float(row["candidate_hit"]) for row in selected])),
                "selection_accuracy_on_hits": float(np.mean([row["selection_correct"] for row in hit_objects])) if hit_objects else None,
                "J": float(np.mean([row["J"] for row in objects])),
                "F": float(np.mean([row["F"] for row in objects])),
                "J_and_F": float(np.mean([row["J_and_F"] for row in objects])),
            })
    return output


def smoke_checks(evaluation: list[dict], metrics: dict, feature_audit: dict) -> dict:
    videos = []
    for row in evaluation:
        if row["video_id"] not in videos:
            videos.append(row["video_id"])
        if len(videos) == 2:
            break
    sample = [row for row in evaluation if row["video_id"] in videos and row["tracks"].shape[0]][:8]
    mapping = all(
        set(map(int, row["candidate_track_ids"])) == set(candidate_lookup(metrics[row["identity"]]["concept"]))
        for row in sample
    )
    shuffled = all(fixed_derangement(row["identity"]) != [0, 1, 2, 3] for row in sample)
    static_temporal_distinct = all(
        not np.allclose(row["tracks"][:, -1:].expand(-1, 4, -1).numpy(), row["tracks"].numpy())
        for row in sample if row["tracks"].shape[0]
    )
    checks = {
        "videos": videos,
        "expressions_checked": len(sample),
        "candidate_identity_mapping_correct": mapping,
        "direct_candidate_metrics_present": all(bool(candidate_lookup(metrics[row["identity"]]["concept"])) for row in sample),
        "ground_truth_entered_feature_or_candidate_generation": bool(feature_audit.get("ground_truth_read", True)),
        "order_shuffle_effective": shuffled,
        "static_temporal_inputs_distinct": static_temporal_distinct,
    }
    checks["pass"] = all(value for key, value in checks.items() if key not in {"videos", "expressions_checked", "ground_truth_entered_feature_or_candidate_generation"}) and not checks["ground_truth_entered_feature_or_candidate_generation"]
    return checks


def run(args) -> int:
    import torch

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status(root, "running", "load_cached_inputs")
    manifest = json.loads(Path(args.manifest).read_text())
    all_manifest = {}
    for item in manifest["objects"]:
        for expression in item["expressions"]:
            identity = "/".join((item["dataset"], item["video_id"], str(item["object_id"]), str(expression["expression_id"])))
            all_manifest[identity] = {
                "identity": identity, "dataset": item["dataset"], "split": item["split"],
                "video_id": item["video_id"], "object_id": str(item["object_id"]),
                "expression_id": str(expression["expression_id"]), "description_type": expression["type"],
                "expression": expression["text"],
            }
    metrics = read_metrics(Path(args.eval_metrics))
    records = read_candidate_records(Path(args.candidate_root))
    train_all, train_hits = examples(Path(args.train_features), Path(args.train_metrics), args.hit_threshold)
    evaluation, _ = examples(Path(args.eval_features), Path(args.eval_metrics), args.hit_threshold)
    evaluation_by_id = {row["identity"]: row for row in evaluation}
    feature_audit = json.loads(Path(args.eval_features).with_suffix(".json").read_text())
    smoke = smoke_checks(evaluation, metrics, feature_audit)
    (root / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if not smoke["pass"]:
        status(root, "failed", "smoke", smoke=smoke)
        raise RuntimeError(f"FTSG smoke failed: {smoke}")

    status(root, "running", "load_frozen_bigrus", smoke=smoke)
    load_args = SimpleNamespace(
        load_checkpoint_dir=args.probe_checkpoints,
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hit_threshold=args.hit_threshold,
        device=args.device,
    )
    models, histories, checkpoint_manifest = load_models(load_args)
    static_predictions = predict_probe(models[False], evaluation, False, args.device)
    torch.cuda.reset_peak_memory_stats(args.device); torch.cuda.synchronize(args.device); static_started=time.monotonic()
    predict_probe(models[False], evaluation, False, args.device)
    torch.cuda.synchronize(args.device); static_elapsed=time.monotonic()-static_started; static_peak=int(torch.cuda.max_memory_allocated(args.device))
    temporal_predictions, _, _ = predict(models[True], evaluation, args.device, "normal")
    _, temporal_elapsed, temporal_peak = predict(models[True], evaluation, args.device, "normal")
    order_predictions, order_elapsed, order_peak = predict(models[True], evaluation, args.device, "order_shuffle")

    fitting, validation = split_train(train_hits)
    mean_models = []
    mean_statuses = []
    status(root, "running", "train_mean_pool_controls", seeds=args.seeds)
    for seed in args.seeds:
        checkpoint = root / "checkpoints" / f"mean_pool_seed{seed}.pt"
        training_status = root / "checkpoints" / f"mean_pool_seed{seed}.json"
        if checkpoint.is_file() and training_status.is_file():
            model, _ = load_mean_pool(checkpoint, args.device)
            value = json.loads(training_status.read_text())
        else:
            model, value = train_mean_pool(
                fitting, validation, seed, args.device, args.epochs, args.patience, args.learning_rate
            )
            save_mean_pool(model, value, checkpoint)
            training_status.write_text(json.dumps(value, indent=2) + "\n")
        mean_models.append(model)
        mean_statuses.append(value)
        status(root, "running", "train_mean_pool_controls", completed_seeds=len(mean_models), seeds=args.seeds)
    mean_predictions, mean_elapsed, mean_peak = predict(mean_models, evaluation, args.device, "normal")

    rows = []
    selection_rows = []
    control_rows = []
    missing_states = set(feature_audit.get("missing_state_identities", []))
    for identity, meta in sorted(all_manifest.items()):
        concept = metrics.get(identity, {}).get("concept")
        raw = metrics.get(identity, {}).get("raw_expression")
        example = evaluation_by_id.get(identity)
        method_selected = {
            "sam31_raw_expression": native_top(raw),
            "sam31_concept_native": native_top(concept),
            "oracle_identity": oracle_id(concept, args.hit_threshold),
        }
        if example is not None:
            method_selected.update({
                "static_identity": selected_from_logits(example, static_predictions.get(identity)),
                "temporal_identity": selected_from_logits(example, temporal_predictions.get(identity)),
                "temporal_order_shuffle": selected_from_logits(example, order_predictions.get(identity)),
                "temporal_mean_pool": selected_from_logits(example, mean_predictions.get(identity)),
            })
        else:
            method_selected.update({name: None for name in ("static_identity", "temporal_identity", "temporal_order_shuffle", "temporal_mean_pool")})
        expression_rows = {}
        for method in METHODS:
            metric = raw if method == "sam31_raw_expression" else concept
            state_failure = identity in missing_states and method in {"static_identity", "temporal_identity", "temporal_order_shuffle", "temporal_mean_pool"}
            native_unavailable = method in {"sam31_raw_expression", "sam31_concept_native"} and bool(candidate_lookup(metric)) and method_selected[method] is None
            row_status = "state_failure" if state_failure else ("native_score_unavailable" if native_unavailable else "success")
            value = selection_row(meta, method, metric, method_selected[method], args.hit_threshold, row_status)
            rows.append(value)
            expression_rows[method] = value
            if method in {"temporal_identity", "temporal_order_shuffle", "temporal_mean_pool"}:
                control_rows.append(value)
        selection_rows.append({
            **meta,
            "concept_candidate_count": expression_rows["temporal_identity"]["candidate_count"],
            "concept_candidate_hit": expression_rows["temporal_identity"]["candidate_hit"],
            "oracle_track_id": expression_rows["oracle_identity"]["selected_track_id"],
            **{f"{method}_selected_track_id": expression_rows[method]["selected_track_id"] for method in METHODS},
            **{f"{method}_selection_correct": expression_rows[method]["selection_correct"] for method in METHODS},
        })

    summary_rows = method_summary(rows)
    comparisons = {}
    for kind in ("static", "dynamic", "hybrid"):
        comparisons[kind] = {
            "temporal_minus_static_accuracy": paired_bootstrap(rows, "temporal_identity", "static_identity", "selection_correct", kind, True),
            "temporal_minus_static_J_and_F": paired_bootstrap(rows, "temporal_identity", "static_identity", "J_and_F", kind),
            "temporal_minus_order_shuffle_accuracy": paired_bootstrap(rows, "temporal_identity", "temporal_order_shuffle", "selection_correct", kind, True),
            "temporal_minus_order_shuffle_J_and_F": paired_bootstrap(rows, "temporal_identity", "temporal_order_shuffle", "J_and_F", kind),
            "temporal_minus_mean_pool_accuracy": paired_bootstrap(rows, "temporal_identity", "temporal_mean_pool", "selection_correct", kind, True),
            "temporal_minus_mean_pool_J_and_F": paired_bootstrap(rows, "temporal_identity", "temporal_mean_pool", "J_and_F", kind),
        }

    concept_record_values = [row for (identity, method), row in records.items() if method == "concept" and identity in all_manifest]
    candidate_latency = float(np.mean([row["latency_seconds_synchronized"] for row in concept_record_values]))
    candidate_peak = max(int(row["peak_memory_bytes"]) for row in concept_record_values)
    selector_queries = len(evaluation)
    efficiency = [
        {"component": "sam31_concept_candidate_generation", "params_per_head": "frozen", "ensemble_size": 1, "params_total_loaded": "frozen", "latency_seconds_per_query": candidate_latency, "peak_memory_bytes": candidate_peak},
        {"component": "static_identity_selector_ensemble", "params_per_head": count_parameters(models[False][0]), "ensemble_size": len(models[False]), "params_total_loaded": count_parameters(models[False][0])*len(models[False]), "latency_seconds_per_query": static_elapsed / selector_queries, "peak_memory_bytes": static_peak},
        {"component": "temporal_identity_selector_ensemble", "params_per_head": count_parameters(models[True][0]), "ensemble_size": len(models[True]), "params_total_loaded": count_parameters(models[True][0])*len(models[True]), "latency_seconds_per_query": temporal_elapsed / selector_queries, "peak_memory_bytes": temporal_peak},
        {"component": "temporal_order_shuffle_ensemble", "params_per_head": count_parameters(models[True][0]), "ensemble_size": len(models[True]), "params_total_loaded": count_parameters(models[True][0])*len(models[True]), "latency_seconds_per_query": order_elapsed / selector_queries, "peak_memory_bytes": order_peak},
        {"component": "temporal_mean_pool_ensemble", "params_per_head": count_parameters(mean_models[0]), "ensemble_size": len(mean_models), "params_total_loaded": count_parameters(mean_models[0])*len(mean_models), "latency_seconds_per_query": mean_elapsed / selector_queries, "peak_memory_bytes": mean_peak},
        {"component": "ftsg_total_candidate_plus_selector", "params_per_head": count_parameters(models[True][0]), "ensemble_size": len(models[True]), "params_total_loaded": count_parameters(models[True][0])*len(models[True]), "latency_seconds_per_query": candidate_latency + temporal_elapsed / selector_queries, "peak_memory_bytes": max(candidate_peak, temporal_peak)},
    ]
    dynamic = comparisons["dynamic"]
    main_go = dynamic["temporal_minus_static_accuracy"]["ci_low"] > 0 and dynamic["temporal_minus_static_J_and_F"]["ci_low"] > 0
    ordered_effects = (
        dynamic["temporal_minus_order_shuffle_J_and_F"],
        dynamic["temporal_minus_mean_pool_J_and_F"],
    )
    order_supported = any(value["mean"] > 0 and value["ci_low"] > 0 for value in ordered_effects)
    decision = "GO" if main_go and order_supported else ("PARTIAL GO" if main_go else "NO-GO")

    write_csv(root / "per_expression_results.csv", rows)
    write_csv(root / "candidate_selection.csv", selection_rows)
    write_csv(root / "temporal_controls.csv", control_rows)
    write_csv(root / "efficiency.csv", efficiency)
    audit = {
        "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": {name: str(Path(getattr(args, name)).resolve()) for name in ("manifest", "train_features", "train_metrics", "eval_features", "eval_metrics", "candidate_root", "probe_checkpoints")},
        "manifest_objects": len(manifest["objects"]), "manifest_expressions": len(all_manifest),
        "evaluated_feature_expressions": len(evaluation), "missing_state_identities": sorted(missing_states),
        "candidate_generation_failures": feature_audit.get("missing_candidate_identities", []),
        "ground_truth_entered_inference": False,
        "frozen_components": ["Sa2VA", "SAM3.1", "visual encoders"],
        "main_checkpoint_manifest": checkpoint_manifest,
        "smoke": smoke,
    }
    (root / "run_config.json").write_text(json.dumps(audit, indent=2) + "\n")
    result = {
        "decision": decision,
        "main_go": main_go,
        "temporal_order_supported": order_supported,
        "method_summary": summary_rows,
        "comparisons": comparisons,
        "efficiency": efficiency,
        "mean_pool_training": mean_statuses,
        "main_training": {"seeds": args.seeds, "checkpoint_mode": "load_only", "histories": histories},
        "coverage": {"objects": len(manifest["objects"]), "expressions": len(all_manifest), "feature_successes": len(evaluation), "state_failures": len(missing_states)},
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    status(root, "complete", "full_validation_complete", decision=decision, coverage=result["coverage"], elapsed_seconds=result["elapsed_seconds"])
    print(json.dumps({"decision": decision, "dynamic": dynamic, "coverage": result["coverage"]}, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--eval-features", required=True)
    parser.add_argument("--eval-metrics", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--probe-checkpoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 42])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

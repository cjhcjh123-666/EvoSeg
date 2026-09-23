"""Summarize within-model Static/Dynamic gaps on one fixed manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SUCCESS = {"success", "success_no_seg"}


def read_jsonl(path: Path) -> list[dict]:
    by_key = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["key"] in by_key:
                    raise RuntimeError(
                        f"duplicate prediction key in {path}: {record['key']}"
                    )
                by_key[record["key"]] = record
    return list(by_key.values())


def read_prediction_files(value: str | list[str]) -> tuple[list[dict], list[str]]:
    paths = [Path(path) for path in value] if isinstance(value, list) else [Path(value)]
    if not paths:
        raise ValueError("predictions file list must not be empty")
    by_key = {}
    for path in paths:
        for record in read_jsonl(path):
            if record["key"] in by_key:
                raise RuntimeError(
                    f"duplicate prediction key across files: {record['key']}"
                )
            by_key[record["key"]] = record
    return list(by_key.values()), [str(path) for path in paths]


def expression_identity(record: dict) -> tuple[str, str, str, str]:
    return (
        record["dataset"],
        record["video_id"],
        str(record["object_id"]),
        str(record["expression_id"]),
    )


def audit_virst_runtime_protocol(records: list[dict]) -> dict:
    """Fail closed if VIRST's realized frame protocol differs within a video.

    The all-object run is sharded by expression, so expressions from one source
    video can execute in different processes and on different GPUs.  The runtime
    audit therefore has to merge all prediction shards before comparing realized
    VLM/SAM indices and the deterministic per-video sampling seed.
    """

    required = {
        "video_id",
        "vlm_frame_indices",
        "segmentation_frame_indices",
        "actual_vlm_frame_count",
        "actual_sam_frame_count",
        "sampling_seed",
        "gt_available_to_model",
    }
    missing = [
        record.get("key", "<missing-key>")
        for record in records
        if not required.issubset(record)
    ]
    if missing:
        raise RuntimeError(
            "VIRST records lack required runtime protocol fields: "
            + ", ".join(missing[:10])
        )

    signatures_by_video = defaultdict(set)
    violations = []
    for record in records:
        vlm_indices = [int(value) for value in record["vlm_frame_indices"]]
        sam_indices = [
            int(value) for value in record["segmentation_frame_indices"]
        ]
        if record["gt_available_to_model"] is not False:
            violations.append(f"{record['key']}: GT available to model")
        if len(vlm_indices) != int(record["actual_vlm_frame_count"]):
            violations.append(f"{record['key']}: VLM frame count mismatch")
        if len(sam_indices) != int(record["actual_sam_frame_count"]):
            violations.append(f"{record['key']}: SAM frame count mismatch")
        if vlm_indices != sorted(vlm_indices):
            violations.append(f"{record['key']}: VLM frames not time ordered")
        if sam_indices != sorted(sam_indices):
            violations.append(f"{record['key']}: SAM frames not time ordered")
        if not set(sam_indices).issubset(vlm_indices):
            violations.append(f"{record['key']}: SAM frames not a VLM subset")
        signatures_by_video[record["video_id"]].add(
            (
                tuple(vlm_indices),
                tuple(sam_indices),
                int(record["sampling_seed"]),
            )
        )
    mismatched_videos = sorted(
        video_id
        for video_id, signatures in signatures_by_video.items()
        if len(signatures) != 1
    )
    if violations or mismatched_videos:
        details = violations[:10]
        if mismatched_videos:
            details.append(
                "same-video signatures differ: " + ", ".join(mismatched_videos[:10])
            )
        raise RuntimeError("VIRST runtime protocol audit failed: " + "; ".join(details))
    return {
        "status": "pass",
        "audited_successful_records": len(records),
        "audited_source_videos": len(signatures_by_video),
        "same_video_sampling_signature_mismatches": 0,
        "gt_available_to_model_records": 0,
        "actual_vlm_frame_counts": sorted(
            {int(record["actual_vlm_frame_count"]) for record in records}
        ),
        "actual_sam_frame_counts": sorted(
            {int(record["actual_sam_frame_count"]) for record in records}
        ),
        "sampling_signature_fields": [
            "vlm_frame_indices",
            "segmentation_frame_indices",
            "sampling_seed",
        ],
    }


def manifest_identities(manifest: dict) -> set[tuple[str, str, str, str]]:
    return {
        (
            item["dataset"],
            item["video_id"],
            str(item["object_id"]),
            str(expression["expression_id"]),
        )
        for item in manifest["objects"]
        for expression in item["expressions"]
    }


def manifest_expressions_by_object(
    manifest: dict,
) -> dict[tuple[str, str, str], dict[str, set]]:
    expected = defaultdict(lambda: defaultdict(set))
    for item in manifest["objects"]:
        object_key = (item["dataset"], item["video_id"], str(item["object_id"]))
        for expression in item["expressions"]:
            expected[object_key][expression["type"]].add(
                (
                    item["dataset"],
                    item["video_id"],
                    str(item["object_id"]),
                    str(expression["expression_id"]),
                )
            )
    return {
        object_key: {kind: set(identities) for kind, identities in by_type.items()}
        for object_key, by_type in expected.items()
    }


def select_manifest_objects(manifest: dict, max_objects: int | None) -> dict:
    if max_objects is None:
        return manifest
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    return {**manifest, "objects": manifest["objects"][:max_objects]}


def build_pair_overlap_audit(
    pairs: list[dict], model_names: list[str], expected_by_object: dict
) -> dict:
    expected_objects = {
        object_key
        for object_key, by_type in expected_by_object.items()
        if by_type.get("static") and by_type.get("dynamic")
    }
    by_model = defaultdict(set)
    for row in pairs:
        by_model[row["model"]].add(
            (row["dataset"], row["video_id"], str(row["object_id"]))
        )
    complete_sets = [by_model[name] for name in model_names]
    shared = set.intersection(*complete_sets) if complete_sets else set()

    def serialize(values):
        return ["/".join(value) for value in sorted(values)]

    return {
        "expected_paired_objects": len(expected_objects),
        "successful_models": model_names,
        "complete_paired_objects_by_model": {
            name: len(by_model[name]) for name in model_names
        },
        "missing_paired_objects_by_model": {
            name: serialize(expected_objects - by_model[name]) for name in model_names
        },
        "shared_complete_paired_objects": len(shared),
        "shared_complete_object_keys": serialize(shared),
        "all_models_have_identical_complete_object_set": all(
            by_model[name] == shared for name in model_names
        ),
    }


def cluster_bootstrap(
    rows: list[dict], iterations: int, seed: int
) -> tuple[float, float, float]:
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row["dynamic_minus_static_J_and_F"])
    videos = sorted(by_video)
    point = float(np.mean([value for values in by_video.values() for value in values]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(iterations):
        selected = rng.choice(videos, size=len(videos), replace=True)
        boot.append(
            float(np.mean([value for video in selected for value in by_video[str(video)]]))
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def summarize_shared_pairs(
    pairs: list[dict], model_names: list[str], iterations: int, seed: int
) -> list[dict]:
    """Summarize every model on the identical all-model-complete object set."""
    by_model = defaultdict(list)
    object_sets = {}
    for row in pairs:
        by_model[row["model"]].append(row)
    for name in model_names:
        object_sets[name] = {
            (row["dataset"], row["video_id"], str(row["object_id"]))
            for row in by_model[name]
        }
    shared = (
        set.intersection(*(object_sets[name] for name in model_names))
        if model_names
        else set()
    )
    summaries = []
    for name in model_names:
        rows = [
            row
            for row in by_model[name]
            if (row["dataset"], row["video_id"], str(row["object_id"])) in shared
        ]
        if not rows:
            summaries.append(
                {
                    "model": name,
                    "status": "unavailable_no_shared_complete_pairs",
                    "shared_paired_objects": 0,
                }
            )
            continue
        point, low, high = cluster_bootstrap(rows, iterations, seed)
        summary = {
            "model": name,
            "status": "success",
            "shared_paired_objects": len(rows),
            "source_videos": len({row["video_id"] for row in rows}),
            "dynamic_minus_static_J_and_F": point,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "dynamic_worse_object_fraction": float(
                np.mean([row["dynamic_worse"] for row in rows])
            ),
            "bootstrap_unit": "source_video",
            "bootstrap_iterations": iterations,
        }
        for description_type in ("static", "dynamic"):
            for metric in ("J", "F", "J_and_F"):
                summary[f"{description_type}_{metric}"] = float(
                    np.mean([row[f"{description_type}_{metric}"] for row in rows])
                )
        summaries.append(summary)
    return summaries


def summarize_model(
    spec: dict,
    expected: set,
    expected_by_object: dict,
    iterations: int,
    seed: int,
):
    records, prediction_files = read_prediction_files(spec["predictions"])
    if spec.get("frame_budget") is not None:
        records = [
            record
            for record in records
            if int(record.get("frame_budget", spec["frame_budget"]))
            == int(spec["frame_budget"])
        ]
    successful_all = [record for record in records if record.get("status") in SUCCESS]
    successful = [
        record
        for record in successful_all
        if expression_identity(record) in expected
    ]
    all_succeeded_identities = {
        expression_identity(record) for record in successful_all
    }
    succeeded_identities = {expression_identity(record) for record in successful}
    failed = [record for record in records if record.get("status") not in SUCCESS]
    by_identity = {}
    for record in successful:
        identity = expression_identity(record)
        if identity in by_identity:
            raise RuntimeError(
                f"duplicate successful expression identity after condition filter for "
                f"{spec['name']}: {'/'.join(identity)}"
            )
        by_identity[identity] = record
    pair_rows = []
    incomplete_objects = []
    eligible_objects = [
        object_key
        for object_key, by_type in expected_by_object.items()
        if by_type.get("static") and by_type.get("dynamic")
    ]
    for dataset, video_id, object_id in sorted(eligible_objects):
        by_type = expected_by_object[(dataset, video_id, object_id)]
        expected_static = by_type["static"]
        expected_dynamic = by_type["dynamic"]
        missing_static = sorted(expected_static - succeeded_identities)
        missing_dynamic = sorted(expected_dynamic - succeeded_identities)
        if missing_static or missing_dynamic:
            incomplete_objects.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "object_id": object_id,
                    "missing_static_expression_ids": [
                        value[3] for value in missing_static
                    ],
                    "missing_dynamic_expression_ids": [
                        value[3] for value in missing_dynamic
                    ],
                }
            )
            continue
        static_records = [by_identity[identity] for identity in sorted(expected_static)]
        dynamic_records = [by_identity[identity] for identity in sorted(expected_dynamic)]
        static = {
            metric: float(np.mean([record[metric] for record in static_records]))
            for metric in ("J", "F", "J_and_F")
        }
        dynamic = {
            metric: float(np.mean([record[metric] for record in dynamic_records]))
            for metric in ("J", "F", "J_and_F")
        }
        pair_rows.append(
            {
                "model": spec["name"],
                "dataset": dataset,
                "video_id": video_id,
                "object_id": object_id,
                "static_expression_count": len(static_records),
                "dynamic_expression_count": len(dynamic_records),
                "static_expression_ids": ";".join(
                    identity[3] for identity in sorted(expected_static)
                ),
                "dynamic_expression_ids": ";".join(
                    identity[3] for identity in sorted(expected_dynamic)
                ),
                **{f"static_{metric}": static[metric] for metric in static},
                **{f"dynamic_{metric}": dynamic[metric] for metric in dynamic},
                "dynamic_minus_static_J_and_F": dynamic["J_and_F"]
                - static["J_and_F"],
                "dynamic_worse": int(dynamic["J_and_F"] < static["J_and_F"]),
            }
        )
    if not pair_rows:
        raise RuntimeError(f"no complete Static/Dynamic object pairs for {spec['name']}")
    point, low, high = cluster_bootstrap(pair_rows, iterations, seed)
    summary = {
        "model": spec["name"],
        "status": "success",
        "paired_objects": len(pair_rows),
        "eligible_paired_objects": len(eligible_objects),
        "excluded_incomplete_paired_objects": len(incomplete_objects),
        "source_videos": len({row["video_id"] for row in pair_rows}),
        "static_J": float(np.mean([row["static_J"] for row in pair_rows])),
        "static_F": float(np.mean([row["static_F"] for row in pair_rows])),
        "static_J_and_F": float(
            np.mean([row["static_J_and_F"] for row in pair_rows])
        ),
        "dynamic_J": float(np.mean([row["dynamic_J"] for row in pair_rows])),
        "dynamic_F": float(np.mean([row["dynamic_F"] for row in pair_rows])),
        "dynamic_J_and_F": float(
            np.mean([row["dynamic_J_and_F"] for row in pair_rows])
        ),
        "dynamic_minus_static_J_and_F": point,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "dynamic_worse_object_fraction": float(
            np.mean([row["dynamic_worse"] for row in pair_rows])
        ),
        "bootstrap_unit": "source_video",
        "bootstrap_iterations": iterations,
        "input_frames": spec.get("input_frames"),
        "resolution": spec.get("resolution"),
        "checkpoint": spec.get("checkpoint"),
        "checkpoint_source": spec.get("checkpoint_source"),
        "training_data": spec.get("training_data"),
        "explicit_temporal_module": spec.get("explicit_temporal_module"),
    }
    audit = {
        "model": spec["name"],
        "prediction_files": prediction_files,
        "records_after_condition_filter": len(records),
        "successful_records": len(successful),
        "failed_records": len(failed),
        "expected_expression_identities": len(expected),
        "successful_expression_identities": len(succeeded_identities),
        "missing_expression_identities": [
            "/".join(value) for value in sorted(expected - succeeded_identities)
        ],
        "extra_expression_identities": [
            "/".join(value) for value in sorted(all_succeeded_identities - expected)
        ],
        "failed_keys": [record["key"] for record in failed],
        "eligible_paired_objects": len(eligible_objects),
        "complete_paired_objects": len(pair_rows),
        "incomplete_paired_objects": incomplete_objects,
    }
    if spec.get("runtime_protocol_audit") == "virst_locked_frames":
        audit["runtime_protocol"] = audit_virst_runtime_protocol(successful)
    return summary, pair_rows, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args) -> None:
    config = json.loads(Path(args.config).read_text())
    full_manifest = json.loads(Path(config["manifest"]).read_text())
    manifest = select_manifest_objects(full_manifest, config.get("max_objects"))
    expected = manifest_identities(manifest)
    expected_by_object = manifest_expressions_by_object(manifest)
    summaries = []
    pairs = []
    audits = []
    for spec in config["models"]:
        if spec.get("status") == "blocked":
            summaries.append(
                {
                    "model": spec["name"],
                    "status": "blocked",
                    "blocker": spec["blocker"],
                    "checkpoint": spec.get("checkpoint"),
                    "checkpoint_source": spec.get("checkpoint_source"),
                    "explicit_temporal_module": spec.get("explicit_temporal_module"),
                }
            )
            audits.append(
                {
                    "model": spec["name"],
                    "status": "blocked",
                    "blocker": spec["blocker"],
                }
            )
            continue
        summary, model_pairs, audit = summarize_model(
            spec, expected, expected_by_object, args.bootstrap_iterations, args.seed
        )
        summaries.append(summary)
        pairs.extend(model_pairs)
        audits.append(audit)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "cross_model_summary.csv", summaries)
    write_csv(output_dir / "cross_model_object_pairs.csv", pairs)
    atomic = output_dir / "cross_model_audit.json"
    atomic.write_text(json.dumps(audits, indent=2, ensure_ascii=False) + "\n")
    successful_model_names = [
        summary["model"] for summary in summaries if summary["status"] == "success"
    ]
    overlap = build_pair_overlap_audit(
        pairs, successful_model_names, expected_by_object
    )
    (output_dir / "cross_model_pair_overlap.json").write_text(
        json.dumps(overlap, indent=2, ensure_ascii=False) + "\n"
    )
    shared_summaries = summarize_shared_pairs(
        pairs,
        successful_model_names,
        args.bootstrap_iterations,
        args.seed,
    )
    write_csv(output_dir / "cross_model_shared_summary.csv", shared_summaries)
    protocol = {
        "manifest": config["manifest"],
        "selection": (
            f"first {config['max_objects']} objects in the prediction-independent "
            "seed-42 manifest order"
            if config.get("max_objects") is not None
            else "all objects in the prediction-independent seed-42 manifest"
        ),
        "max_objects": config.get("max_objects"),
        "full_manifest_objects": len(full_manifest["objects"]),
        "selected_objects": len(manifest["objects"]),
        "expected_expression_identities": len(expected),
        "bootstrap_unit": "source_video",
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
    }
    (output_dir / "cross_model_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

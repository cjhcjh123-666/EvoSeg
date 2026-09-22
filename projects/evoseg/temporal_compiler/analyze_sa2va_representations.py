"""Analyze Sa2VA representation movement against cached segmentation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.stats import spearmanr


DESCRIPTION_TYPES = ("static", "dynamic", "hybrid")
BUDGET_PAIRS = ((8, 16), (16, 32), (8, 32))
SPACES = ("h_seg", "z_seg")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"invalid vector norm: {norm}")
    return vector / norm


def distances(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_unit = unit(left)
    right_unit = unit(right)
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return {
        "cosine_similarity": cosine,
        "angular_distance_radians": float(math.acos(cosine)),
        # Explicit definition: Euclidean distance after independently
        # normalizing both vectors to unit L2 norm.
        "normalized_l2": float(np.linalg.norm(left_unit - right_unit)),
    }


def vector_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    return unit(np.mean([unit(vector) for vector in vectors], axis=0))


def load_vectors(run_dir: Path, records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    vectors = {}
    for record in records:
        if record.get("status") != "success":
            continue
        value = np.load(run_dir / record["vector_path"])
        vectors[record["key"]] = {
            space: np.asarray(value[space]).reshape(-1) for space in SPACES
        }
    return vectors


def decode_masks(path: Path) -> tuple[list[str], np.ndarray]:
    encoded = json.loads(path.read_text())
    names = [item["frame_name"] for item in encoded]
    masks = []
    for item in encoded:
        rle = dict(item["rle"])
        rle["counts"] = rle["counts"].encode("ascii")
        masks.append(mask_utils.decode(rle).astype(bool))
    return names, np.stack(masks)


def compare_masks(path8: Path, path32: Path) -> dict[str, float]:
    names8, masks8 = decode_masks(path8)
    names32, masks32 = decode_masks(path32)
    if names8 != names32 or masks8.shape != masks32.shape:
        raise AssertionError(
            f"cached output frames differ: {path8}, {path32}, "
            f"names_equal={names8 == names32}, shapes={masks8.shape}/{masks32.shape}"
        )
    intersection = np.logical_and(masks8, masks32).sum(axis=(1, 2))
    union = np.logical_or(masks8, masks32).sum(axis=(1, 2))
    prediction_iou = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )
    pixels = masks8.shape[-2] * masks8.shape[-1]
    area8 = masks8.sum(axis=(1, 2)) / pixels
    area32 = masks32.sum(axis=(1, 2)) / pixels
    disagreement = np.logical_xor(masks8, masks32).sum(axis=(1, 2)) / pixels
    return {
        "prediction_iou_n8_n32": float(np.mean(prediction_iou)),
        "signed_mask_area_fraction_change_n32_minus_n8": float(
            np.mean(area32 - area8)
        ),
        "absolute_mask_area_fraction_change_n8_n32": float(
            np.mean(np.abs(area32 - area8))
        ),
        "framewise_pixel_disagreement_n8_n32": float(np.mean(disagreement)),
    }


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def cluster_bootstrap_spearman(
    rows: list[dict], x_field: str, y_field: str, seed: int, iterations: int
) -> dict:
    usable = [
        row
        for row in rows
        if np.isfinite(row[x_field]) and np.isfinite(row[y_field])
    ]
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in usable:
        clusters[row["video_id"]].append(row)
    names = sorted(clusters)
    if len(usable) < 3 or len(names) < 2:
        return {
            "n": len(usable),
            "source_videos": len(names),
            "rho": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "valid_bootstrap_iterations": 0,
        }
    observed = float(spearmanr([row[x_field] for row in usable], [row[y_field] for row in usable]).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(iterations):
        selected = rng.choice(names, size=len(names), replace=True)
        sample = [row for name in selected for row in clusters[str(name)]]
        rho = spearmanr(
            [row[x_field] for row in sample], [row[y_field] for row in sample]
        ).statistic
        if np.isfinite(rho):
            boot.append(float(rho))
    return {
        "n": len(usable),
        "source_videos": len(names),
        "rho": observed,
        "ci_low": float(np.quantile(boot, 0.025)) if boot else math.nan,
        "ci_high": float(np.quantile(boot, 0.975)) if boot else math.nan,
        "valid_bootstrap_iterations": len(boot),
    }


def analyze(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    previous_dir = Path(args.previous_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    records = load_jsonl(run_dir / "representation_records.jsonl")
    failed = [record for record in records if record.get("status") != "success"]
    if failed:
        raise RuntimeError(f"representation extraction contains {len(failed)} failures")
    by_key = {record["key"]: record for record in records}
    if len(by_key) != len(records):
        raise AssertionError("duplicate representation keys found")
    previous_records = load_jsonl(previous_dir / "predictions.jsonl")
    previous = {record["key"]: record for record in previous_records}
    vectors = load_vectors(run_dir, records)
    if set(by_key) != set(vectors):
        raise AssertionError("successful record/vector key mismatch")

    identity_groups: dict[tuple[str, str, str, str], dict[int, dict]] = defaultdict(dict)
    for record in records:
        identity = (
            record["dataset"],
            record["video_id"],
            record["object_id"],
            record["expression_id"],
        )
        identity_groups[identity][int(record["frame_budget"])] = record

    movement_rows = []
    relation_rows = []
    for identity, budget_records in sorted(identity_groups.items()):
        for low, high in BUDGET_PAIRS:
            if low not in budget_records or high not in budget_records:
                continue
            left = budget_records[low]
            right = budget_records[high]
            for space in SPACES:
                movement_rows.append(
                    {
                        "dataset": identity[0],
                        "video_id": identity[1],
                        "object_id": identity[2],
                        "expression_id": identity[3],
                        "description_type": left["description_type"],
                        "expression": left["expression"],
                        "space": space,
                        "budget_low": low,
                        "budget_high": high,
                        **distances(vectors[left["key"]][space], vectors[right["key"]][space]),
                    }
                )
        if 8 not in budget_records or 32 not in budget_records:
            continue
        record8 = budget_records[8]
        record32 = budget_records[32]
        prior8 = previous[record8["key"]]
        prior32 = previous[record32["key"]]
        relation = {
            "dataset": identity[0],
            "video_id": identity[1],
            "object_id": identity[2],
            "expression_id": identity[3],
            "description_type": record8["description_type"],
            "expression": record8["expression"],
            "jf_n8": prior8["J_and_F"],
            "jf_n32": prior32["J_and_F"],
            "delta_jf_n32_minus_n8": prior32["J_and_F"] - prior8["J_and_F"],
        }
        for space in SPACES:
            for metric, value in distances(
                vectors[record8["key"]][space], vectors[record32["key"]][space]
            ).items():
                relation[f"{space}_{metric}"] = value
        relation.update(
            compare_masks(
                previous_dir / prior8["rle_masks_path"],
                previous_dir / prior32["rle_masks_path"],
            )
        )
        relation_rows.append(relation)

    movement_summary = []
    for description_type in DESCRIPTION_TYPES:
        for low, high in BUDGET_PAIRS:
            for space in SPACES:
                selected = [
                    row
                    for row in movement_rows
                    if row["description_type"] == description_type
                    and row["budget_low"] == low
                    and row["budget_high"] == high
                    and row["space"] == space
                ]
                for metric in (
                    "cosine_similarity",
                    "angular_distance_radians",
                    "normalized_l2",
                ):
                    movement_summary.append(
                        {
                            "analysis": "same_expression_frame_budget",
                            "description_type": description_type,
                            "budget_pair": f"{low}_vs_{high}",
                            "space": space,
                            "metric": metric,
                            **finite_summary([row[metric] for row in selected]),
                        }
                    )

    # Calibration B: one type-centroid comparison per object.  This prevents
    # objects with more official expressions from receiving extra weight.
    at16: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record["frame_budget"] == 16:
            at16[(record["dataset"], record["video_id"], record["object_id"])][
                record["description_type"]
            ].append(record)
    calibration_rows = []
    for identity, by_type in sorted(at16.items()):
        if by_type.get("static") and by_type.get("dynamic"):
            for space in SPACES:
                static = vector_centroid(
                    [vectors[record["key"]][space] for record in by_type["static"]]
                )
                dynamic = vector_centroid(
                    [vectors[record["key"]][space] for record in by_type["dynamic"]]
                )
                calibration_rows.append(
                    {
                        "comparison": "same_object_static_vs_dynamic_n16",
                        "dataset": identity[0],
                        "video_id": identity[1],
                        "object_id_a": identity[2],
                        "object_id_b": identity[2],
                        "space": space,
                        **distances(static, dynamic),
                    }
                )

    # Calibration C: compare type-balanced expression centroids for distinct
    # objects in the same video.  All inputs are official expressions.
    video_objects: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for identity, by_type in sorted(at16.items()):
        per_space = {}
        for space in SPACES:
            type_centroids = [
                vector_centroid([vectors[record["key"]][space] for record in values])
                for values in by_type.values()
                if values
            ]
            per_space[space] = vector_centroid(type_centroids)
        video_objects[(identity[0], identity[1])][identity[2]] = per_space
    for (dataset, video_id), objects in sorted(video_objects.items()):
        object_ids = sorted(objects)
        for index, object_a in enumerate(object_ids):
            for object_b in object_ids[index + 1 :]:
                for space in SPACES:
                    calibration_rows.append(
                        {
                            "comparison": "same_video_different_object_n16",
                            "dataset": dataset,
                            "video_id": video_id,
                            "object_id_a": object_a,
                            "object_id_b": object_b,
                            "space": space,
                            **distances(
                                objects[object_a][space], objects[object_b][space]
                            ),
                        }
                    )

    calibration_summary = []
    # A comes directly from same-expression 8-vs-32 movements.
    for description_type in DESCRIPTION_TYPES:
        for space in SPACES:
            selected = [
                row
                for row in movement_rows
                if row["description_type"] == description_type
                and row["budget_low"] == 8
                and row["budget_high"] == 32
                and row["space"] == space
            ]
            for metric in (
                "cosine_similarity",
                "angular_distance_radians",
                "normalized_l2",
            ):
                calibration_summary.append(
                    {
                        "comparison": "same_expression_n8_vs_n32",
                        "description_type": description_type,
                        "space": space,
                        "metric": metric,
                        "aggregation_unit": "expression",
                        **finite_summary([row[metric] for row in selected]),
                    }
                )
    for comparison in (
        "same_object_static_vs_dynamic_n16",
        "same_video_different_object_n16",
    ):
        for space in SPACES:
            selected = [
                row
                for row in calibration_rows
                if row["comparison"] == comparison and row["space"] == space
            ]
            if comparison == "same_video_different_object_n16":
                per_video = defaultdict(list)
                for row in selected:
                    per_video[row["video_id"]].append(row)
            for metric in (
                "cosine_similarity",
                "angular_distance_radians",
                "normalized_l2",
            ):
                if comparison == "same_video_different_object_n16":
                    values = [
                        float(np.mean([row[metric] for row in rows]))
                        for rows in per_video.values()
                    ]
                    unit_name = "source_video"
                else:
                    values = [row[metric] for row in selected]
                    unit_name = "object"
                calibration_summary.append(
                    {
                        "comparison": comparison,
                        "description_type": "all_official",
                        "space": space,
                        "metric": metric,
                        "aggregation_unit": unit_name,
                        **finite_summary(values),
                    }
                )

    correlation_rows = []
    for description_type in DESCRIPTION_TYPES:
        selected = [
            row for row in relation_rows if row["description_type"] == description_type
        ]
        for space in SPACES:
            for metric in (
                "angular_distance_radians",
                "normalized_l2",
            ):
                result = cluster_bootstrap_spearman(
                    selected,
                    f"{space}_{metric}",
                    "delta_jf_n32_minus_n8",
                    seed=args.seed,
                    iterations=args.bootstrap_iterations,
                )
                correlation_rows.append(
                    {
                        "description_type": description_type,
                        "space": space,
                        "representation_metric": metric,
                        "outcome": "delta_jf_n32_minus_n8",
                        "bootstrap_unit": "source_video",
                        "bootstrap_iterations": args.bootstrap_iterations,
                        **result,
                    }
                )

    mask_summary = []
    for description_type in DESCRIPTION_TYPES:
        selected = [
            row for row in relation_rows if row["description_type"] == description_type
        ]
        for metric in (
            "prediction_iou_n8_n32",
            "signed_mask_area_fraction_change_n32_minus_n8",
            "absolute_mask_area_fraction_change_n8_n32",
            "framewise_pixel_disagreement_n8_n32",
            "delta_jf_n32_minus_n8",
        ):
            mask_summary.append(
                {
                    "description_type": description_type,
                    "metric": metric,
                    **finite_summary([row[metric] for row in selected]),
                }
            )

    write_csv(output_dir / "representation_movements.csv", movement_rows)
    write_csv(output_dir / "representation_summary.csv", movement_summary)
    write_csv(output_dir / "representation_calibration.csv", calibration_rows)
    write_csv(output_dir / "representation_calibration_summary.csv", calibration_summary)
    write_csv(output_dir / "representation_vs_jf.csv", relation_rows)
    write_csv(output_dir / "representation_jf_correlations.csv", correlation_rows)
    write_csv(output_dir / "prediction_mask_change_summary.csv", mask_summary)
    audit = {
        "representation_records": len(records),
        "unique_representation_keys": len(by_key),
        "successful_representation_records": len(vectors),
        "failed_representation_records": len(failed),
        "unique_expressions_with_n8_n32": len(relation_rows),
        "seg_token_count_values": sorted(
            {record["token_info"]["seg_token_count"] for record in records}
        ),
        "segmentation_invoked_values": sorted(
            {record["segmentation_invoked"] for record in records}
        ),
        "ground_truth_loaded_values": sorted(
            {record["ground_truth_loaded"] for record in records}
        ),
        "normalized_l2_definition": "L2 distance between independently unit-normalized vectors",
        "calibration_b_unit": "object; official expressions averaged within type first",
        "calibration_c_unit": "source video; official expressions type-balanced within object",
        "bootstrap_seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    (output_dir / "representation_analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--previous-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

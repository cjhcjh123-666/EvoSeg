"""Execute fixed candidate-derived point plans through official SAM3.1."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    audit_loaded_checkpoint,
    sha256,
)
from projects.evoseg.temporal_grounding_interface.run_dynamic_interface import (
    infer_condition,
    tracking_signature,
)
from projects.evoseg.temporal_grounding_mechanism.common import identity
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_gt(item: dict, shape: tuple[int, int]) -> np.ndarray:
    result = []
    for path_value, present in zip(item["evaluation_mask_paths"], item["evaluation_mask_present"]):
        path = Path(path_value)
        if bool(present) != path.is_file():
            raise RuntimeError(f"GT availability changed: {path}")
        if present:
            with Image.open(path) as image:
                result.append(np.asarray(image.convert("L")) > 0)
        else:
            result.append(np.zeros(shape, dtype=bool))
    return np.stack(result)


def rebuild_tracking_cache(plans: list[dict], successful_records: dict) -> dict[str, dict]:
    """Rebuild exact public-prompt cache after a resumable process restart."""
    plan_by_key = {(row["identity"], row["condition"]): row for row in plans}
    cache = {}
    for key, previous in successful_records.items():
        plan_row = plan_by_key.get(key)
        if plan_row is None:
            continue
        signature = (
            f"{plan_row['object_id']}:"
            + tracking_signature(plan_row["video_id"], plan_row["plan"])
        )
        cache[signature] = {
            value: previous[value]
            for value in (
                "status",
                "J",
                "F",
                "J_and_F",
                "prompt_updates",
                "latency_seconds_synchronized",
                "peak_memory_bytes",
                "session_api_compat",
            )
        }
        cache[signature]["reused_source_identity"] = previous["identity"]
        cache[signature]["reused_source_condition"] = previous["condition"]
    return cache


def run(args) -> int:
    sam3_repo = Path(args.sam3_repo).resolve()
    sys.path.insert(0, str(sam3_repo))
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    source = Path(inspect.getfile(build_sam3_multiplex_video_predictor)).resolve()
    if sam3_repo not in source.parents:
        raise RuntimeError(f"SAM3 imported outside official checkout: {source}")
    checkpoint = Path(args.checkpoint).resolve()
    if sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    manifest = json.loads(Path(args.manifest).read_text())
    item_index = {
        (item["dataset"], item["video_id"], str(item["object_id"])): item
        for item in manifest["objects"][: args.max_objects]
    }
    plans = []
    with Path(args.plans).open() as handle:
        for line in handle:
            row = json.loads(line)
            object_position = next(
                index
                for index, item in enumerate(manifest["objects"][: args.max_objects])
                if item["video_id"] == row["video_id"] and str(item["object_id"]) == str(row["object_id"])
            )
            if object_position % args.num_shards == args.shard_index:
                plans.append(row)
    output_dir = Path(args.run_dir).resolve()
    records_path = output_dir / "point_results.jsonl"
    successful_records = {}
    if records_path.is_file():
        with records_path.open() as handle:
            successful_records = {
                (row["identity"], row["condition"]): row
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
                if row.get("status") == "success"
            }
    done = set(successful_records)
    torch.cuda.set_device(args.device)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    torch.cuda.synchronize()
    audit = audit_loaded_checkpoint(predictor.model, checkpoint)
    atomic(
        output_dir / "run_config.json",
        {
            "command": [sys.executable, *sys.argv],
            "adapter_source": str(Path(__file__).resolve()),
            "adapter_source_sha256": sha256(Path(__file__).resolve()),
            "official_sam3_repo": str(sam3_repo),
            "official_sam3_commit": __import__("subprocess").check_output(["git", "-C", str(sam3_repo), "rev-parse", "HEAD"], text=True).strip(),
            "runtime_builder_source": str(source),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "weight_audit": audit,
            "protocol": {
                "ground_truth_entered_model": False,
                "point_source": "deterministic interior point of the selected cached candidate track mask",
                "tracker": "official public add_prompt plus bidirectional propagation",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        },
    )
    attempted = failed = 0
    # Different official expressions for one object often induce an identical
    # fixed candidate/point plan.  Reuse only the evaluated result for that
    # exact object and exact plan; masks are not retained and no prediction is
    # shared across objects.
    tracking_cache: dict[str, dict] = rebuild_tracking_cache(plans, successful_records)
    started = time.monotonic()
    image_root = Path(manifest["dataset"]["image_root"])
    for row in plans:
        key = (row["identity"], row["condition"])
        if key in done:
            continue
        item = item_index[(row["dataset"], row["video_id"], str(row["object_id"]))]
        result = {key: row[key] for key in ("identity", "dataset", "video_id", "object_id", "expression_id", "description_type", "expression", "condition", "ORACLE")}
        result["gt_entered_model"] = False
        result["started_at"] = time.time()
        try:
            cache_key = (
                f"{row['object_id']}:"
                + tracking_signature(row["video_id"], row["plan"])
            )
            if cache_key in tracking_cache:
                result.update(tracking_cache[cache_key], cache_hit=True)
                result["reused_tracking_signature"] = cache_key
                result["selected_track_ids"] = [
                    value["selected_candidate_object_id"] for value in row["plan"]
                ]
                done.add(key)
                append(records_path, result)
                attempted += 1
                elapsed = time.monotonic() - started
                atomic(
                    output_dir / "STATUS.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "planned": len(plans),
                        "completed": len(done),
                        "attempted_this_process": attempted,
                        "failed_this_process": failed,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / attempted * max(len(plans) - len(done), 0),
                        "current": row["identity"] + "/" + row["condition"],
                    },
                )
                continue
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            then = time.perf_counter()
            inference = infer_condition(
                predictor,
                image_root / item["video_id"],
                item,
                None,
                None,
                None,
                selection_plan=row["plan"],
            )
            torch.cuda.synchronize()
            with Image.open(image_root / item["video_id"] / f"{item['frame_names'][0]}.jpg") as image:
                shape = (image.height, image.width)
            prediction = np.stack([inference["masks"][index] for index in item["evaluation_frame_indices"]])
            gt = load_gt(item, shape)
            j_values = db_eval_iou(gt, prediction)
            f_values = db_eval_boundary(gt, prediction)
            j = float(np.mean(j_values))
            f = float(np.mean(f_values))
            result.update(
                status="success",
                J=j,
                F=f,
                J_and_F=(j + f) / 2,
                selected_track_ids=[value["selected_candidate_object_id"] for value in row["plan"]],
                prompt_updates=inference["update_count"],
                latency_seconds_synchronized=time.perf_counter() - then,
                peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                session_api_compat=inference["session_api_compat"],
                cache_hit=False,
                reused_tracking_signature=None,
                reused_source_identity=None,
                reused_source_condition=None,
            )
            tracking_cache[cache_key] = {
                key: result[key]
                for key in (
                    "status",
                    "J",
                    "F",
                    "J_and_F",
                    "prompt_updates",
                    "latency_seconds_synchronized",
                    "peak_memory_bytes",
                    "session_api_compat",
                )
            }
            tracking_cache[cache_key]["reused_source_identity"] = result["identity"]
            tracking_cache[cache_key]["reused_source_condition"] = result["condition"]
            done.add(key)
        except Exception as error:
            failed += 1
            result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        append(records_path, result)
        attempted += 1
        elapsed = time.monotonic() - started
        atomic(
            output_dir / "STATUS.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "planned": len(plans),
                "completed": len(done),
                "attempted_this_process": attempted,
                "failed_this_process": failed,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / attempted * max(len(plans) - len(done), 0),
                "current": row["identity"] + "/" + row["condition"],
            },
        )
    atomic(
        output_dir / "STATUS.json",
        {
            "state": "complete" if not failed else "complete_with_failures",
            "pid": os.getpid(),
            "planned": len(plans),
            "completed": len(done),
            "attempted_this_process": attempted,
            "failed_this_process": failed,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_remaining_seconds": 0.0,
            "current": None,
        },
    )
    return 0 if not failed else 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

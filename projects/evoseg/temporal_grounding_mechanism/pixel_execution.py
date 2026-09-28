"""Decompose candidate-mask compression and SAM3.1 video propagation loss.

Candidate identities and stage plans are fixed before this runner starts.  GT is
opened only after the public prompt call has produced an initialization mask.
"""

from __future__ import annotations

import argparse
import csv
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
    start_multiplex_session,
)
from projects.evoseg.temporal_grounding_interface.protocol import mask_iou
from projects.evoseg.temporal_grounding_interface.run_dynamic_interface import (
    decode_rle,
    infer_condition,
    selected_object_mask,
    stream_masks,
)
from projects.evoseg.temporal_grounding_mechanism.common import identity
from projects.evoseg.temporal_grounding_mechanism.point_executor import load_gt
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


BASES = {
    "PREDICTED_CANDIDATE_POINT_STATIC": "predicted_static",
    "ORACLE_ID_POINT": "oracle_identity",
}


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


def metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    j = float(np.mean(db_eval_iou(gt, prediction)))
    f = float(np.mean(db_eval_boundary(gt, prediction)))
    return {"J": j, "F": f, "J_and_F": (j + f) / 2}


def candidate_sources(root: Path) -> dict[str, Path]:
    result = {}
    shards = sorted(root.glob("shard-*")) or [root]
    for shard in shards:
        records = shard / "candidate_records.jsonl"
        if not records.is_file():
            continue
        with records.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "success" or row.get("prompt_method") != "concept":
                    continue
                key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                result[key] = shard / row["candidate_tracks_path"]
    return result


def load_plans(path: Path, manifest: dict, max_objects: int, shard: int, shards: int) -> list[dict]:
    object_index = {
        (item["video_id"], str(item["object_id"])): index
        for index, item in enumerate(manifest["objects"][:max_objects])
    }
    result = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] not in BASES:
                continue
            object_key = (row["video_id"], str(row["object_id"]))
            if object_key not in object_index:
                continue
            index = object_index[object_key]
            if index % shards == shard:
                result.append(row)
    return result


def candidate_stage_masks(tracks: dict, plan: list[dict]) -> list[np.ndarray]:
    evaluation = [int(value) for value in tracks["evaluation_frame_indices"]]
    result = []
    for stage in plan:
        track_id = stage["selected_candidate_object_id"]
        position = evaluation.index(int(stage["anchor_frame_index"]))
        track = next(
            (value for value in tracks["tracks"] if int(value["track_id"]) == int(track_id)),
            None,
        ) if track_id is not None else None
        if track is None:
            result.append(np.zeros((int(tracks["height"]), int(tracks["width"])), dtype=bool))
        else:
            result.append(decode_rle(track["frames"][position]))
    return result


def tight_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    return [
        float(xs.min() / width),
        float(ys.min() / height),
        float((xs.max() - xs.min() + 1) / width),
        float((ys.max() - ys.min() + 1) / height),
    ]


def best_output(outputs: dict, reference: np.ndarray) -> tuple[int | None, np.ndarray]:
    ids = [int(value) for value in np.asarray(outputs["out_obj_ids"]).reshape(-1)]
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    if not ids:
        return None, np.zeros_like(reference)
    selected = int(np.argmax([mask_iou(reference, value) for value in masks]))
    return ids[selected], masks[selected]


def infer_box(predictor, video_path: Path, item: dict, plan: list[dict], candidate_masks: list[np.ndarray]) -> dict:
    shape = candidate_masks[0].shape
    session, compatibility = start_multiplex_session(predictor, str(video_path))
    session_id = session["session_id"]
    final_masks: dict[int, np.ndarray] = {}
    stages = []
    previous_endpoint = -1
    try:
        for selection, candidate_mask in zip(plan, candidate_masks):
            endpoint = int(selection["anchor_frame_index"])
            chunk_start = previous_endpoint + 1
            box = tight_box(candidate_mask)
            post_update = np.zeros(shape, dtype=bool)
            output_id = None
            if box is not None:
                response = predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=endpoint,
                    bounding_boxes=[box],
                    bounding_box_labels=[1],
                    rel_coordinates=True,
                )
                output_id, post_update = best_output(response["outputs"], candidate_mask)
            if output_id is None:
                backward = {}
            else:
                backward = stream_masks(
                    predictor, session_id, "backward", endpoint,
                    endpoint - chunk_start + 1, output_id, shape,
                )
            backward[endpoint] = post_update
            for index in range(chunk_start, endpoint + 1):
                final_masks[index] = backward.get(index, np.zeros(shape, dtype=bool))
            stages.append({**selection, "post_update_mask": post_update, "box_xywh": box, "sam_output_object_id": output_id})
            previous_endpoint = endpoint
    finally:
        predictor.close_session(session_id)
    missing = set(range(item["frame_count"])) - set(final_masks)
    if missing:
        raise RuntimeError(f"box condition did not cover {len(missing)} frames")
    return {"masks": final_masks, "stages": stages, "session_api_compat": compatibility}


def infer_point_initialization(predictor, video_path: Path, plan: list[dict], shape: tuple[int, int]) -> dict:
    """Measure each point as a fresh object slot, without any propagation."""
    session, compatibility = start_multiplex_session(predictor, str(video_path))
    session_id = session["session_id"]
    stages = []
    try:
        for selection in plan:
            object_id = int(selection["stage_order"]) + 1
            point = selection["positive_point_relative_xy"]
            initialized = np.zeros(shape, dtype=bool)
            output_ids = []
            output_areas = []
            if point is not None:
                response = predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=int(selection["anchor_frame_index"]),
                    points=[list(point)], point_labels=[1], obj_id=object_id,
                    rel_coordinates=True,
                )
                output_ids = [int(value) for value in np.asarray(response["outputs"]["out_obj_ids"]).reshape(-1)]
                output_areas = [int(np.asarray(value, dtype=bool).sum()) for value in np.asarray(response["outputs"]["out_binary_masks"])]
                initialized = selected_object_mask(response["outputs"], object_id, shape)
            stages.append({
                **selection, "post_update_mask": initialized,
                "requested_sam_object_id": object_id, "sam_output_object_ids": output_ids,
                "sam_output_mask_areas": output_areas,
            })
    finally:
        predictor.close_session(session_id)
    return {"stages": stages, "session_api_compat": compatibility}


def load_anchor_gt(item: dict, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    parent = Path(item["evaluation_mask_paths"][0]).parent
    path = parent / f"{item['frame_names'][frame_index]}.png"
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def plan_signature(
    row: dict, prompt_form: str, candidate_masks: list[np.ndarray] | None = None
) -> str:
    if prompt_form == "box" and candidate_masks is None:
        raise ValueError("box execution signature requires the candidate masks")
    values = [
        {
            "frame": stage["anchor_frame_index"],
            "track": stage["selected_candidate_object_id"],
            "point": stage["positive_point_relative_xy"] if prompt_form == "point" else None,
            "box": tight_box(candidate_masks[index]) if prompt_form == "box" else None,
        }
        for index, stage in enumerate(row["plan"])
    ]
    return json.dumps([row["video_id"], row["object_id"], prompt_form, values], sort_keys=True)


def load_prior_point(root: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted(root.glob("point_*_[0-9]/point_results.jsonl")) + sorted(root.glob("point_shard_*/point_results.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success":
                    result[(row["identity"], row["condition"])] = row
    return result


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
    items = {
        (item["video_id"], str(item["object_id"])): item
        for item in manifest["objects"][: args.max_objects]
    }
    plans = load_plans(Path(args.plans), manifest, args.max_objects, args.shard_index, args.num_shards)
    if not 0 <= args.plan_slice_index < args.plan_slice_count:
        raise ValueError("plan-slice-index must be in [0, plan-slice-count)")
    plans = plans[args.plan_slice_index :: args.plan_slice_count]
    if args.max_plans is not None:
        plans = plans[: args.max_plans]
    candidates = candidate_sources(Path(args.candidate_root))
    prior = load_prior_point(Path(args.prior_point_root))
    output = Path(args.run_dir).resolve()
    records_path = output / "pixel_execution_results.jsonl"
    completed = {}
    valid_keys = {
        (row["identity"], BASES[row["condition"]], prompt_form)
        for row in plans
        for prompt_form in ("point", "box")
    }
    resume_paths = [records_path, *(Path(value) for value in args.resume_records)]
    for resume_path in resume_paths:
        if not resume_path.is_file():
            continue
        with resume_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["identity"], row["identity_basis"], row["prompt_form"])
                if row.get("status") == "success" and key in valid_keys:
                    completed[key] = row
    done = set(completed)
    result_cache = {
        row["execution_signature"]: row
        for row in completed.values()
        if row.get("execution_signature")
    }

    torch.cuda.set_device(args.device)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint), max_num_objects=16, multiplex_count=16,
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False,
    )
    torch.cuda.synchronize()
    audit = audit_loaded_checkpoint(predictor.model, checkpoint)
    atomic(output / "run_config.json", {
        "command": [sys.executable, *sys.argv],
        "official_sam3_repo": str(sam3_repo),
        "official_sam3_commit": __import__("subprocess").check_output(["git", "-C", str(sam3_repo), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_builder_source": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "weight_audit": audit,
        "public_api": {
            "point": "add_prompt(points, point_labels, obj_id)",
            "box": "add_prompt(bounding_boxes, bounding_box_labels)",
            "mask": "N/A for multiplex: public add_mask rejects models without add_tracker_new_mask",
        },
        "protocol": {
            "candidate_identity_fixed_before_execution": True,
            "gt_entered_model": False,
            "box_output_identity": "highest candidate-mask IoU among public box outputs; no GT",
            "box_stage_semantics": "official box add_prompt resets semantic state at every stage",
        },
    })
    attempted = failed = 0
    started = time.monotonic()
    image_root = Path(manifest["dataset"]["image_root"])
    for row in plans:
        item = items[(row["video_id"], str(row["object_id"]))]
        track_path = candidates[row["identity"]]
        tracks = json.loads(track_path.read_text())
        candidate_masks = candidate_stage_masks(tracks, row["plan"])
        gt = load_gt(item, candidate_masks[0].shape)
        for prompt_form in ("point", "box"):
            basis = BASES[row["condition"]]
            key = (row["identity"], basis, prompt_form)
            if key in done:
                continue
            result = {
                key: row[key]
                for key in ("identity", "dataset", "video_id", "object_id", "expression_id", "description_type", "expression", "ORACLE")
            }
            signature = plan_signature(row, prompt_form, candidate_masks)
            result.update(identity_basis=basis, prompt_form=prompt_form, execution_signature=signature, gt_entered_model=False)
            try:
                if signature in result_cache:
                    cached = result_cache[signature]
                    for name in (
                        "anchor_candidate_J", "anchor_candidate_F", "anchor_candidate_J_and_F",
                        "init_J", "init_F", "init_J_and_F", "candidate_to_init_IoU",
                        "propagated_J", "propagated_F", "propagated_J_and_F",
                        "initialization_loss_J_and_F", "propagation_loss_J_and_F",
                        "stage_metrics", "latency_seconds_synchronized", "peak_memory_bytes",
                        "prior_point_propagated_J_and_F", "rerun_minus_prior_J_and_F",
                    ):
                        result[name] = cached.get(name)
                    result.update(status="success", cache_hit=True, reused_source_identity=cached["identity"])
                else:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    then = time.perf_counter()
                    previous = prior.get((row["identity"], row["condition"]))
                    if prompt_form == "point":
                        if previous is None:
                            raise RuntimeError("missing completed prior C2 point propagation")
                        inference = infer_point_initialization(
                            predictor, image_root / item["video_id"], row["plan"],
                            candidate_masks[0].shape,
                        )
                    else:
                        inference = infer_box(
                            predictor, image_root / item["video_id"], item,
                            row["plan"], candidate_masks,
                        )
                    torch.cuda.synchronize()
                    stage_rows = []
                    for selection, candidate_mask, stage in zip(row["plan"], candidate_masks, inference["stages"]):
                        frame = int(selection["anchor_frame_index"])
                        anchor_gt = load_anchor_gt(item, frame, candidate_mask.shape)
                        initialized = stage["post_update_mask"]
                        candidate_score = metrics(anchor_gt[None], candidate_mask[None])
                        init_score = metrics(anchor_gt[None], initialized[None])
                        stage_rows.append({
                            "stage_order": selection["stage_order"],
                            "canonical_stage_index": selection["canonical_stage_index"],
                            "anchor_frame_index": frame,
                            "selected_candidate_object_id": selection["selected_candidate_object_id"],
                            "candidate_J": candidate_score["J"],
                            "candidate_F": candidate_score["F"],
                            "candidate_J_and_F": candidate_score["J_and_F"],
                            "init_J": init_score["J"],
                            "init_F": init_score["F"],
                            "init_J_and_F": init_score["J_and_F"],
                            "candidate_to_init_IoU": mask_iou(candidate_mask, initialized),
                            "box_xywh": stage.get("box_xywh"),
                            "requested_sam_object_id": stage.get("requested_sam_object_id"),
                            "sam_output_object_ids": stage.get("sam_output_object_ids"),
                            "sam_output_mask_areas": stage.get("sam_output_mask_areas"),
                            "sam_output_object_id": stage.get("sam_output_object_id"),
                        })
                    if prompt_form == "point":
                        propagated = {
                            "J": float(previous["J"]), "F": float(previous["F"]),
                            "J_and_F": float(previous["J_and_F"]),
                        }
                    else:
                        prediction = np.stack([inference["masks"][int(index)] for index in item["evaluation_frame_indices"]])
                        propagated = metrics(gt, prediction)
                    means = {
                        name: float(np.mean([stage[name] for stage in stage_rows]))
                        for name in ("candidate_J", "candidate_F", "candidate_J_and_F", "init_J", "init_F", "init_J_and_F", "candidate_to_init_IoU")
                    }
                    result.update(
                        status="success", cache_hit=False, reused_source_identity=None,
                        anchor_candidate_J=means["candidate_J"],
                        anchor_candidate_F=means["candidate_F"],
                        anchor_candidate_J_and_F=means["candidate_J_and_F"],
                        init_J=means["init_J"], init_F=means["init_F"], init_J_and_F=means["init_J_and_F"],
                        candidate_to_init_IoU=means["candidate_to_init_IoU"],
                        propagated_J=propagated["J"], propagated_F=propagated["F"], propagated_J_and_F=propagated["J_and_F"],
                        initialization_loss_J_and_F=means["init_J_and_F"] - means["candidate_J_and_F"],
                        propagation_loss_J_and_F=propagated["J_and_F"] - means["init_J_and_F"],
                        stage_metrics=stage_rows,
                        latency_seconds_synchronized=time.perf_counter() - then,
                        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                    )
                    if prompt_form == "point":
                        result["prior_point_propagated_J_and_F"] = previous["J_and_F"]
                        result["rerun_minus_prior_J_and_F"] = 0.0
                    else:
                        result["prior_point_propagated_J_and_F"] = None
                        result["rerun_minus_prior_J_and_F"] = None
                    result_cache[signature] = result
                done.add(key)
            except Exception as error:
                failed += 1
                result.update(status="failed", error=str(error), traceback=traceback.format_exc())
            append(records_path, result)
            attempted += 1
            elapsed = time.monotonic() - started
            atomic(output / "STATUS.json", {
                "state": "running", "pid": os.getpid(), "planned": len(plans) * 2,
                "completed": len(done), "attempted_this_process": attempted,
                "failed_this_process": failed, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / attempted * max(len(plans) * 2 - len(done), 0),
                "current": f"{row['identity']}/{basis}/{prompt_form}",
            })
    atomic(output / "STATUS.json", {
        "state": "complete" if not failed else "complete_with_failures", "pid": os.getpid(),
        "planned": len(plans) * 2, "completed": len(done), "attempted_this_process": attempted,
        "failed_this_process": failed, "elapsed_seconds": time.monotonic() - started,
        "estimated_remaining_seconds": 0.0, "current": None,
    })
    return 0 if not failed else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--prior-point-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--plan-slice-index", type=int, default=0)
    parser.add_argument("--plan-slice-count", type=int, default=1)
    parser.add_argument("--resume-records", nargs="*", default=[])
    parser.add_argument("--max-plans", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

"""Run official SAM 3.1 tracking with stage-wise public point corrections.

Candidate selection uses cached frozen Sa2VA anchor grounding and cached
current-frame SAM3.1 candidates.  ``infer_condition`` has no ground-truth
argument.  GT is opened only after masks and selections have been fixed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    audit_loaded_checkpoint,
    sha256,
    start_multiplex_session,
)
from projects.evoseg.temporal_grounding_interface.protocol import (
    CONDITIONS,
    deterministic_positive_point,
    mask_iou,
    select_candidate,
    stable_identity,
)
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def decode_rle(value: dict) -> np.ndarray:
    value = dict(value)
    value["counts"] = value["counts"].encode("ascii")
    return mask_utils.decode(value).astype(bool)


def encode_rle(value: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(np.asarray(value, dtype=np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def successful_index(path: Path) -> dict[str, dict]:
    result = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("status") == "success":
                if value["identity"] in result:
                    raise RuntimeError(f"duplicate identity: {value['identity']}")
                result[value["identity"]] = value
    return result


def completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            (value["identity"], value["condition"])
            for line in handle
            if line.strip()
            for value in [json.loads(line)]
            if value.get("status") == "success"
        }


def stage_indices_for_count(count: int) -> list[int]:
    if count == 1:
        return [7]
    stride = 8 // count
    return list(range(stride - 1, 8, stride))


def selected_object_mask(outputs: dict, object_id: int, shape: tuple[int, int]) -> np.ndarray:
    ids = [int(value) for value in np.asarray(outputs["out_obj_ids"]).reshape(-1)]
    if object_id not in ids:
        return np.zeros(shape, dtype=bool)
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    return masks[ids.index(object_id)]


def stream_masks(
    predictor,
    session_id: str,
    direction: str,
    start: int,
    maximum: int,
    object_id: int,
    shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    result = {}
    for response in predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction=direction,
        start_frame_idx=start,
        max_frame_num_to_track=maximum,
    ):
        result[int(response["frame_index"])] = selected_object_mask(
            response["outputs"], object_id, shape
        )
    return result


def build_selection_plan(condition, states: np.lib.npyio.NpzFile, candidates: dict) -> list[dict]:
    """Fix all candidate choices before tracker execution, without GT."""

    stage_numbers = stage_indices_for_count(condition.stages)
    stage_by_number = {int(value["stage_index"]): value for value in candidates["stages"]}
    plan = []
    for order, stage_number in enumerate(stage_numbers):
        stage = stage_by_number[stage_number]
        state_kind = (
            "temporal" if condition.state_kind in {"temporal", "global"} else "static"
        )
        reference = np.asarray(states[f"{state_kind}_{stage_number}_mask"], dtype=bool)
        decoded = [decode_rle(value["mask"]) for value in stage["candidates"]]
        selected = select_candidate(reference, decoded)
        selected_candidate = stage["candidates"][selected] if selected is not None else None
        point = deterministic_positive_point(decoded[selected]) if selected is not None else None
        plan.append(
            {
                "stage_order": order,
                "canonical_stage_index": stage_number,
                "anchor_frame_index": int(stage["anchor_frame_index"]),
                "state_kind": state_kind,
                "candidate_count": len(decoded),
                "selected_candidate_index": selected,
                "selected_candidate_object_id": selected_candidate["object_id"]
                if selected_candidate
                else None,
                "selected_candidate_confidence": selected_candidate["confidence"]
                if selected_candidate
                else None,
                "candidate_scores": [mask_iou(reference, value) for value in decoded],
                "update_applied": selected is not None,
                "positive_point_relative_xy": point,
            }
        )
    return plan


def tracking_signature(video_id: str, plan: list[dict]) -> str:
    """The SAM tracker depends only on video, endpoints, and public point prompts."""

    value = [
        {
            "frame": row["anchor_frame_index"],
            "point": row["positive_point_relative_xy"],
            "update": row["update_applied"],
        }
        for row in plan
    ]
    return f"{video_id}:" + json.dumps(value, sort_keys=True, separators=(",", ":"))


def reuse_tracking_output(cached: dict, plan: list[dict]) -> dict:
    if len(cached["stages"]) != len(plan):
        raise RuntimeError("cached tracking stage count differs from selection plan")
    stages = []
    for selection, tracking in zip(plan, cached["stages"]):
        if selection["anchor_frame_index"] != tracking["anchor_frame_index"]:
            raise RuntimeError("cached tracking endpoint differs")
        stages.append(
            {
                **selection,
                "pre_update_mask": tracking["pre_update_mask"],
                "post_update_mask": tracking["post_update_mask"],
            }
        )
    return {
        "masks": cached["masks"],
        "stages": stages,
        "session_api_compat": cached["session_api_compat"],
        "update_count": sum(value["update_applied"] for value in plan),
        "vlm_forward_count": len(plan),
    }


def infer_condition(
    predictor,
    video_path: Path,
    item: dict,
    condition,
    states: np.lib.npyio.NpzFile,
    candidates: dict,
    selection_plan: list[dict] | None = None,
) -> dict:
    """Inference-only function: by design, it cannot receive ground truth."""

    plan = selection_plan or build_selection_plan(condition, states, candidates)
    with Image.open(video_path / f"{item['frame_names'][0]}.jpg") as image:
        shape = (image.height, image.width)
    session, compatibility = start_multiplex_session(predictor, str(video_path))
    session_id = session["session_id"]
    maintained_object_id = 1
    final_masks: dict[int, np.ndarray] = {}
    maintained_masks: dict[int, np.ndarray] = {}
    track_started = False
    stages = []
    previous_endpoint = -1
    try:
        for selection in plan:
            endpoint = int(selection["anchor_frame_index"])
            chunk_start = previous_endpoint + 1
            pre_update = maintained_masks.get(endpoint) if track_started else None

            update_applied = bool(selection["update_applied"])
            point = selection["positive_point_relative_xy"]
            post_update = None
            if update_applied:
                response = predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=endpoint,
                    points=[list(point)],
                    point_labels=[1],
                    obj_id=maintained_object_id,
                    rel_coordinates=True,
                )
                post_update = selected_object_mask(
                    response["outputs"], maintained_object_id, shape
                )
                backward = stream_masks(
                    predictor,
                    session_id,
                    "backward",
                    endpoint,
                    endpoint - chunk_start + 1,
                    maintained_object_id,
                    shape,
                )
                backward[endpoint] = post_update
                # The public multiplex API expects the two directions belonging
                # to one correction to be propagated consecutively.  Materialize
                # the forward continuation now and retain it as the pre-update
                # hypothesis for the next stage.  This avoids invoking full-VG
                # propagation before the first object point has established a
                # track, and does not access private SAM state.
                forward = stream_masks(
                    predictor,
                    session_id,
                    "forward",
                    endpoint,
                    item["frame_count"] - endpoint,
                    maintained_object_id,
                    shape,
                )
                maintained_masks.update(backward)
                maintained_masks.update(forward)
                maintained_masks[endpoint] = post_update
                track_started = True
                for index, mask in backward.items():
                    if chunk_start <= index <= endpoint:
                        final_masks[index] = mask
            for index in range(chunk_start, endpoint + 1):
                if index not in final_masks:
                    final_masks[index] = maintained_masks.get(
                        index, np.zeros(shape, dtype=bool)
                    )
            stages.append(
                {
                    **selection,
                    "pre_update_mask": pre_update,
                    "post_update_mask": post_update,
                }
            )
            previous_endpoint = endpoint
    finally:
        predictor.close_session(session_id)
    missing = set(range(item["frame_count"])) - set(final_masks)
    if missing:
        raise RuntimeError(f"condition did not cover {len(missing)} video frames")
    return {
        "masks": final_masks,
        "stages": stages,
        "session_api_compat": compatibility,
        "update_count": sum(value["update_applied"] for value in plan),
        "vlm_forward_count": len(stages),
    }


def load_eval_gt(item: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for path_value, present in zip(
        item["evaluation_mask_paths"], item["evaluation_mask_present"]
    ):
        path = Path(path_value)
        if bool(present) != path.is_file():
            raise RuntimeError(f"GT availability changed: {path}")
        if path.is_file():
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        else:
            mask = np.zeros(shape, dtype=bool)
        masks.append(mask)
    return np.stack(masks)


def load_frame_gt(item: dict, frame_index: int, shape: tuple[int, int]) -> np.ndarray:
    evaluation_parent = Path(item["evaluation_mask_paths"][0]).parent
    path = evaluation_parent / f"{item['frame_names'][frame_index]}.png"
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def evaluate_fixed_output(inference: dict, item: dict, candidate_value: dict) -> tuple[dict, list[dict]]:
    first_mask = next(iter(inference["masks"].values()))
    shape = first_mask.shape
    evaluation_masks = np.stack(
        [inference["masks"][int(index)] for index in item["evaluation_frame_indices"]]
    )
    gt = load_eval_gt(item, shape)
    j_values = db_eval_iou(gt, evaluation_masks)
    f_values = db_eval_boundary(gt, evaluation_masks)
    stages = []
    candidate_by_stage = {
        int(value["stage_index"]): value for value in candidate_value["stages"]
    }
    for stage in inference["stages"]:
        endpoint = int(stage["anchor_frame_index"])
        stage_candidates = candidate_by_stage[stage["canonical_stage_index"]]["candidates"]
        decoded = [decode_rle(value["mask"]) for value in stage_candidates]
        stage_gt = load_frame_gt(item, endpoint, shape)
        candidate_gt_iou = [mask_iou(stage_gt, value) for value in decoded]
        oracle = int(np.argmax(candidate_gt_iou)) if candidate_gt_iou else None
        selected = stage["selected_candidate_index"]
        oracle_iou = candidate_gt_iou[oracle] if oracle is not None else 0.0
        selected_iou = candidate_gt_iou[selected] if selected is not None else 0.0
        pre_jf = None
        if stage["pre_update_mask"] is not None:
            pre_j = float(db_eval_iou(stage_gt[None], stage["pre_update_mask"][None])[0])
            pre_f = float(db_eval_boundary(stage_gt[None], stage["pre_update_mask"][None])[0])
            pre_jf = (pre_j + pre_f) / 2
        post_jf = None
        if stage["post_update_mask"] is not None:
            post_j = float(db_eval_iou(stage_gt[None], stage["post_update_mask"][None])[0])
            post_f = float(db_eval_boundary(stage_gt[None], stage["post_update_mask"][None])[0])
            post_jf = (post_j + post_f) / 2
        stage_public = {
            key: value
            for key, value in stage.items()
            if key not in {"pre_update_mask", "post_update_mask"}
        }
        stage_public.update(
            {
                "candidate_gt_iou": candidate_gt_iou,
                "oracle_candidate_index": oracle,
                "oracle_candidate_iou": oracle_iou,
                "selected_candidate_iou": selected_iou,
                "candidate_hit_at_0_3": bool(oracle_iou >= 0.3),
                "selection_correct": bool(
                    oracle is not None and oracle_iou >= 0.3 and selected == oracle
                ),
                "pre_update_anchor_J_and_F": pre_jf,
                "post_update_anchor_J_and_F": post_jf,
                "update_delta_anchor_J_and_F": (
                    post_jf - pre_jf if pre_jf is not None and post_jf is not None else None
                ),
            }
        )
        stages.append(stage_public)
    j_score = float(np.mean(j_values))
    f_score = float(np.mean(f_values))
    return {
        "J": j_score,
        "F": f_score,
        "J_and_F": (j_score + f_score) / 2,
        "per_frame_J": j_values.tolist(),
        "per_frame_F": f_values.tolist(),
    }, stages


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
    run_dir = Path(args.run_dir).resolve()
    state_root = Path(args.state_root).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    requested_conditions = args.condition or [value.name for value in CONDITIONS]
    condition_index = {value.name: value for value in CONDITIONS}
    unknown_conditions = sorted(set(requested_conditions) - set(condition_index))
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {unknown_conditions}")
    if len(requested_conditions) != len(set(requested_conditions)):
        raise ValueError("duplicate --condition values")
    selected_conditions = [condition_index[name] for name in requested_conditions]
    items = manifest["objects"][: args.max_objects] if args.max_objects else manifest["objects"]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid object shard")
    items = [
        item
        for object_index, item in enumerate(items)
        if object_index % args.num_shards == args.shard_index
    ]
    item_by_identity = {}
    expression_by_identity = {}
    ordered_identities = []
    for item in items:
        for expression in item["expressions"]:
            identity = stable_identity(
                item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]
            )
            item_by_identity[identity] = item
            expression_by_identity[identity] = expression
            ordered_identities.append(identity)
    if args.max_expressions:
        ordered_identities = ordered_identities[: args.max_expressions]
    states = successful_index(state_root / "stage_grounding_records.jsonl")
    candidates = successful_index(candidate_root / "stage_candidate_records.jsonl")
    missing = [identity for identity in ordered_identities if identity not in states or identity not in candidates]
    if missing:
        raise RuntimeError(f"missing stage cache for {len(missing)} expressions; first={missing[0]}")

    torch.cuda.set_device(args.device)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint),
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        use_fa3=args.use_fa3,
        compile=args.compile,
        warm_up=False,
        async_loading_frames=False,
    )
    torch.cuda.synchronize()
    weight_audit = audit_loaded_checkpoint(predictor.model, checkpoint)
    atomic_json(
        run_dir / "dynamic_interface_config.json",
        {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "evoseg_commit": subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "state_root": str(state_root),
            "candidate_root": str(candidate_root),
            "official_sam3_repo": str(sam3_repo),
            "official_sam3_commit": subprocess.check_output(
                ["git", "-C", str(sam3_repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_builder_source": str(source),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "weight_audit": weight_audit,
            "conditions": [value.__dict__ for value in selected_conditions],
            "ground_truth_passed_to_inference": False,
            "candidate_selection_trainable_parameters": 0,
            "candidate_selection": "IoU(frozen Sa2VA anchor grounding, current SAM3.1 candidate mask)",
            "dynamic_update_api": "public Sam3BasePredictor.add_prompt point refinement with fixed obj_id",
            "stage_chunk_policy": "correct at observed stage endpoint, backward propagate within that stage only",
            "tracking_cache": (
                "reuse only when video, stage endpoints, update flags, and exact "
                "relative point coordinates are identical; reported latency is "
                "the synchronized standalone execution value from the cache fill"
            ),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )
    records_path = run_dir / "dynamic_predictions.jsonl"
    stages_path = run_dir / "dynamic_stages.jsonl"
    done = completed(records_path)
    planned = len(ordered_identities) * len(selected_conditions)
    attempted = failed = 0
    started = time.monotonic()
    tracking_cache = {}
    image_root = Path(manifest["dataset"]["image_root"])
    (run_dir / "prediction_masks").mkdir(parents=True, exist_ok=True)
    for identity in ordered_identities:
        item = item_by_identity[identity]
        expression = expression_by_identity[identity]
        state_record = states[identity]
        candidate_record = candidates[identity]
        candidate_value = json.loads(
            (candidate_root / candidate_record["candidate_path"]).read_text()
        )
        with np.load(state_root / state_record["state_path"]) as state_payload:
            for condition in selected_conditions:
                key = (identity, condition.name)
                if key in done:
                    continue
                record = {
                    "identity": identity,
                    "condition": condition.name,
                    "dataset": item["dataset"],
                    "video_id": item["video_id"],
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "evaluation_frame_indices": item["evaluation_frame_indices"],
                    "gt_entered_model": False,
                    "started_at": utc_now(),
                }
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    condition_started = time.perf_counter()
                    selection_plan = build_selection_plan(
                        condition, state_payload, candidate_value
                    )
                    signature = tracking_signature(item["video_id"], selection_plan)
                    cache_hit = signature in tracking_cache
                    if cache_hit:
                        cached = tracking_cache[signature]
                        inference = reuse_tracking_output(cached["inference"], selection_plan)
                        sam31_latency = cached["latency"]
                        sam_peak = cached["peak_memory"]
                    else:
                        inference = infer_condition(
                            predictor,
                            image_root / item["video_id"],
                            item,
                            condition,
                            state_payload,
                            candidate_value,
                            selection_plan=selection_plan,
                        )
                        torch.cuda.synchronize()
                        sam31_latency = time.perf_counter() - condition_started
                        sam_peak = int(torch.cuda.max_memory_allocated())
                        tracking_cache[signature] = {
                            "inference": inference,
                            "latency": sam31_latency,
                            "peak_memory": sam_peak,
                        }
                    metrics, stage_rows = evaluate_fixed_output(inference, item, candidate_value)
                    selected_stage_indices = {
                        int(value["canonical_stage_index"]) for value in stage_rows
                    }
                    state_kind = (
                        "temporal"
                        if condition.state_kind in {"temporal", "global"}
                        else "static"
                    )
                    selected_state_metadata = [
                        value
                        for value in state_record["state_metadata"]
                        if value["state_kind"] == state_kind
                        and int(value["stage_index"]) in selected_stage_indices
                    ]
                    if len(selected_state_metadata) != condition.stages:
                        raise RuntimeError("selected VLM stage metadata is incomplete")
                    vlm_latency = float(
                        sum(value["latency_seconds_synchronized"] for value in selected_state_metadata)
                    )
                    candidate_latency = float(
                        sum(
                            value.get("latency_seconds_synchronized", 0.0)
                            for value in candidate_value["stages"]
                            if int(value["stage_index"]) in selected_stage_indices
                        )
                    )
                    state_peak = max(
                        int(value["peak_memory_bytes"]) for value in selected_state_metadata
                    )
                    candidate_peak = int(candidate_record["peak_memory_bytes"])
                    relative = Path("prediction_masks") / (
                        f"{identity.replace('/', '__')}__{condition.name}.json"
                    )
                    atomic_json(
                        run_dir / relative,
                        [
                            {
                                "frame_index": int(index),
                                "frame_name": item["frame_names"][int(index)],
                                "rle": encode_rle(inference["masks"][int(index)]),
                            }
                            for index in item["evaluation_frame_indices"]
                        ],
                    )
                    for stage in stage_rows:
                        append_jsonl(
                            stages_path,
                            {
                                "identity": identity,
                                "condition": condition.name,
                                "dataset": item["dataset"],
                                "video_id": item["video_id"],
                                "object_id": item["object_id"],
                                "expression_id": expression["expression_id"],
                                "description_type": expression["type"],
                                **stage,
                            },
                        )
                    record.update(
                        {
                            "status": "success",
                            "completed_at": utc_now(),
                            **metrics,
                            "prompt_updates": inference["update_count"],
                            "vlm_forward_count": inference["vlm_forward_count"],
                            "vlm_latency_seconds_synchronized": vlm_latency,
                            "candidate_latency_seconds_synchronized": candidate_latency,
                            "sam31_latency_seconds_synchronized": sam31_latency,
                            "sam31_tracking_cache_hit": cache_hit,
                            "sam31_tracking_signature": signature,
                            "latency_seconds_synchronized": vlm_latency
                            + candidate_latency
                            + sam31_latency,
                            "peak_memory_bytes": max(sam_peak, state_peak, candidate_peak),
                            "component_peak_memory_bytes": {
                                "vlm": state_peak,
                                "candidate_generator": candidate_peak,
                                "sam31_tracking": sam_peak,
                            },
                            "prediction_masks_path": str(relative),
                            "session_api_compat": inference["session_api_compat"],
                        }
                    )
                    done.add(key)
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    failed += 1
                    record.update(status="failed_oom", error=str(error), traceback=traceback.format_exc())
                except Exception as error:
                    failed += 1
                    record.update(status="failed", error=str(error), traceback=traceback.format_exc())
                append_jsonl(records_path, record)
                attempted += 1
                elapsed = time.monotonic() - started
                atomic_json(
                    run_dir / "DYNAMIC_STATUS.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "planned": planned,
                        "completed": len(done),
                        "attempted_this_process": attempted,
                        "failed_this_process": failed,
                        "current_key": [identity, condition.name],
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / attempted * max(planned - len(done), 0),
                        "updated_at": utc_now(),
                    },
                )
    predictor.shutdown()
    atomic_json(
        run_dir / "DYNAMIC_STATUS.json",
        {
            "state": "complete" if failed == 0 else "complete_with_failures",
            "pid": os.getpid(),
            "planned": planned,
            "completed": len(done),
            "attempted_this_process": attempted,
            "failed_this_process": failed,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_remaining_seconds": 0.0,
            "updated_at": utc_now(),
        },
    )
    return 0 if failed == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-expressions", type=int)
    parser.add_argument(
        "--condition",
        action="append",
        choices=[value.name for value in CONDITIONS],
        help="run only the named condition; repeat for a deterministic scheduling partition",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

"""Evaluate Qwen frame grounding as an automatic SAM3.1 tracker initializer.

The experiment is deliberately training-free.  A trained Frame Prompt model
selects an anchor frame by its own native objectness and writes its predicted
mask into the official tracker memory.  The frozen tracker then carries that
single object identity in both temporal directions.  Ground truth is used only
after inference for metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

from projects.evoseg.ftg.metrics import evaluate_masks
from projects.evoseg.ftg.model import FTGQwenSAM31
from projects.evoseg.ftg.public_video_data import PublicVideoPilotDataset


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame_model(args: argparse.Namespace, device: torch.device) -> FTGQwenSAM31:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "frame_prompt":
        raise ValueError("Qwen-seeded tracking requires a frame_prompt checkpoint")
    result = checkpoint["result"]
    if result.get("sam_interface") != "native_text_residual":
        raise ValueError("expected the scaled native-text Frame Prompt checkpoint")
    if result.get("query_score_mode") != "native":
        raise ValueError("anchor selection requires native SAM query scores")
    model = FTGQwenSAM31(
        args.qwen_checkpoint,
        args.sam_checkpoint,
        args.sam_repo,
        variant="frame_prompt",
        qwen_pixels=args.qwen_pixels,
    ).to(device)
    set_peft_model_state_dict(model.qwen, checkpoint["qwen_lora"])
    model.grounding.load_state_dict(checkpoint["grounding"])
    model.native_residual_scale.data.copy_(
        checkpoint["native_residual_scale"].to(
            device=model.native_residual_scale.device,
            dtype=model.native_residual_scale.dtype,
        )
    )
    model.sam_interface = result["sam_interface"]
    model.query_score_mode = result["query_score_mode"]
    model.executor.visual_chunk_size = args.sam_visual_chunk_size
    model.executor.decode_chunk_size = args.sam_decode_chunk_size
    model.eval()
    return model


@torch.inference_mode()
def _frame_observations(
    model: FTGQwenSAM31, frames: list, query: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted logits and native confidence without touching targets."""
    prompts, _ = model.encode_qwen(frames, query)
    visual = model.executor.extract_grounding_features(frames, prompts.device)
    native_text = model.executor.extract_native_text_features(
        query, len(frames), prompts.device
    )
    all_logits, scores = model.executor.decode_grounding_prompts(
        prompts,
        visual,
        native_text_features=native_text,
        return_all_queries=True,
    )
    selected = scores.argmax(dim=-1)
    rows = torch.arange(len(frames), device=scores.device)
    return all_logits[rows, selected], scores[rows, selected]


def _resize_observations(logits: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    resized = F.interpolate(
        logits[:, None].float(), size=shape, mode="bilinear", align_corners=False
    )[:, 0]
    return (resized.sigmoid() >= 0.5).cpu().numpy()


def _interior_point(mask: np.ndarray, fallback_logits: torch.Tensor) -> tuple[float, float]:
    """Pick the most interior positive point, or the logit maximum if empty."""
    height, width = mask.shape
    if mask.any():
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    else:
        resized = F.interpolate(
            fallback_logits[None, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        y, x = np.unravel_index(int(resized.argmax().item()), (height, width))
    # The official tracker consumes relative xy coordinates.
    return ((x + 0.5) / width, (y + 0.5) / height)


def _track_mask(output: dict, object_id: int, shape: tuple[int, int]) -> np.ndarray:
    ids = np.asarray(output["out_obj_ids"]).tolist()
    if object_id not in ids:
        return np.zeros(shape, dtype=bool)
    return np.asarray(output["out_binary_masks"][ids.index(object_id)], dtype=bool)


@torch.inference_mode()
def _add_predicted_mask(
    tracker,
    state: dict,
    frame_index: int,
    mask: np.ndarray,
    object_id: int,
) -> tuple[int, dict]:
    """Register one Qwen-predicted mask in the official multiplex tracker."""
    metadata = state["tracker_metadata"]
    if not metadata:
        metadata.update(tracker._initialize_metadata())
    tracker._prepare_backbone_feats(state, frame_index, reverse=False)
    object_rank = tracker._assign_new_det_to_gpus(
        new_det_num=1,
        prev_workload_per_gpu=metadata["num_obj_per_gpu"],
    )[0]
    if object_rank != tracker.rank:
        raise RuntimeError("single-GPU tracker assigned the seed to another rank")
    tracker_state = tracker._init_new_sam2_state(state)
    state["sam2_inference_states"].append(tracker_state)
    metadata["obj_ids_per_gpu"][object_rank] = np.concatenate(
        [metadata["obj_ids_per_gpu"][object_rank], np.array([object_id], np.int64)]
    )
    metadata["num_obj_per_gpu"][object_rank] = len(
        metadata["obj_ids_per_gpu"][object_rank]
    )
    metadata["obj_ids_all_gpu"] = np.concatenate(metadata["obj_ids_per_gpu"])
    metadata["max_obj_id"] = max(metadata["max_obj_id"], object_id)
    metadata["obj_id_to_score"][object_id] = 1.0
    metadata["obj_id_to_sam2_score_frame_wise"][frame_index][object_id] = (
        torch.tensor(1.0, dtype=torch.float32, device=tracker.device)
    )
    tracker.add_action_history(
        state, "add", frame_idx=frame_index, obj_ids=[object_id]
    )
    _, object_ids, _, video_masks = tracker.tracker.add_new_masks(
        inference_state=tracker_state,
        frame_idx=frame_index,
        obj_ids=[object_id],
        masks=torch.from_numpy(mask.astype(np.float32))[None],
        add_mask_to_memory=True,
    )
    tracker.tracker.propagate_in_video_preflight(
        tracker_state, run_mem_encoder=True
    )
    if object_id not in object_ids:
        raise RuntimeError("official tracker did not retain the seeded object")
    seeded_mask = (video_masks[object_ids.index(object_id)] > 0.0).to(torch.bool)
    object_to_mask = {object_id: seeded_mask}
    tracker._cache_frame_outputs(state, frame_index, object_to_mask)
    out = {
        "obj_id_to_mask": object_to_mask,
        "obj_id_to_score": metadata["obj_id_to_score"],
        "obj_id_to_sam2_score": metadata["obj_id_to_sam2_score_frame_wise"][
            frame_index
        ],
    }
    return frame_index, tracker._postprocess_output(state, out)


@torch.inference_mode()
def _directional_track(
    tracker,
    frames: list,
    anchor_index: int,
    anchor_mask: np.ndarray,
    reverse: bool,
    object_id: int = 1,
) -> dict[int, dict]:
    state = tracker.init_state(
        resource_path=frames,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_cv2=False,
    )
    added = _add_predicted_mask(
        tracker, state, anchor_index, anchor_mask, object_id
    )
    outputs: dict[int, dict] = {}
    if isinstance(added, tuple) and len(added) == 2 and added[1] is not None:
        outputs[int(added[0])] = added[1]
    for frame_index, output in tracker.propagate_in_video(
        state,
        start_frame_idx=anchor_index,
        max_frame_num_to_track=len(frames),
        reverse=reverse,
        is_last_batch=True,
    ):
        outputs[int(frame_index)] = output
    return outputs


@torch.inference_mode()
def _bidirectional_track(
    tracker,
    frames: list,
    anchor_index: int,
    anchor_mask: np.ndarray,
    object_id: int = 1,
) -> np.ndarray:
    # Independent states avoid an official action-history edge case: after an
    # endpoint anchor, the second direction is classified as a cache fetch even
    # though those frames have not been propagated yet.
    # Masklet confirmation belongs to autonomous detector proposals.  A visual
    # point seed is already an explicit object assertion; applying that filter
    # hides its propagated masks because the interactive path does not grow the
    # detector confirmation arrays.
    confirmation_enabled = getattr(tracker, "masklet_confirmation_enable", None)
    if confirmation_enabled is not None:
        tracker.masklet_confirmation_enable = False
    try:
        outputs = _directional_track(
            tracker, frames, anchor_index, anchor_mask, reverse=False,
            object_id=object_id,
        )
        if anchor_index > 0:
            outputs.update(
                _directional_track(
                    tracker, frames, anchor_index, anchor_mask, reverse=True,
                    object_id=object_id,
                )
            )
    finally:
        if confirmation_enabled is not None:
            tracker.masklet_confirmation_enable = confirmation_enabled
    shape = (frames[0].height, frames[0].width)
    return np.stack(
        [
            _track_mask(outputs.get(index, {"out_obj_ids": [], "out_binary_masks": []}), object_id, shape)
            for index in range(len(frames))
        ]
    )


def _aggregate(records: list[dict], prefix: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        groups[record["dataset"]].append(record)
        groups[f'{record["dataset"]}/{record["expression_type"]}'].append(record)
    metric_names = ("j", "f", "jf", "present_jf", "false_accept", "false_reject")
    return {
        group: {
            "count": len(values),
            **{
                metric: float(np.mean([value[f"{prefix}_{metric}"] for value in values]))
                for metric in metric_names
            },
        }
        for group, values in sorted(groups.items())
    }


def run(args: argparse.Namespace) -> dict:
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)
    manifest = json.loads(args.manifest.read_text())
    dataset = PublicVideoPilotDataset(manifest, "validation")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    dataset.records = dataset.records[args.shard_index :: args.num_shards]
    if args.max_records is not None:
        dataset.records = dataset.records[: args.max_records]
    model = _load_frame_model(args, device)
    records = []
    started = time.time()
    status = {
        "schema_version": 1,
        "stage": "qwen_seeded_sam31_tracker",
        "status": "running",
        "pid": os.getpid(),
        "sample_count": len(dataset),
        "completed_samples": 0,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "ground_truth_used_for_initialization_or_selection": False,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    _atomic_json(args.output / "STATUS.json", status)

    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["masks"]
        shape = tuple(target.shape[-2:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            observation_logits, objectness = _frame_observations(
                model, sample["frames"], sample["expression"]
            )
        observation = _resize_observations(observation_logits, shape)
        anchor_index = int(objectness.float().argmax().item())
        point = _interior_point(
            observation[anchor_index], observation_logits[anchor_index]
        )
        tracked = _bidirectional_track(
            model.executor.assembled,
            sample["frames"],
            anchor_index,
            observation[anchor_index],
        )
        observation_metrics = evaluate_masks(target, observation)
        tracker_metrics = evaluate_masks(target, tracked)
        record = {
            "dataset": sample["dataset"],
            "video_id": sample["video_id"],
            "expression_id": sample["expression_id"],
            "expression_type": sample["expression_type"],
            "expression": sample["expression"],
            "anchor_index": anchor_index,
            "anchor_point_xy": point,
            "anchor_objectness": float(objectness[anchor_index].float().item()),
            "tracked_present_fraction": float(tracked.any(axis=(-2, -1)).mean()),
            **{f"frame_{key}": value for key, value in observation_metrics.items()},
            **{f"tracker_{key}": value for key, value in tracker_metrics.items()},
        }
        records.append(record)
        elapsed = time.time() - started
        status.update(
            {
                "completed_samples": index + 1,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / (index + 1) * (len(dataset) - index - 1),
                "last_record": {
                    "dataset": sample["dataset"],
                    "video_id": sample["video_id"],
                    "frame_jf": record["frame_jf"],
                    "tracker_jf": record["tracker_jf"],
                },
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
        )
        _atomic_json(args.output / "STATUS.json", status)
        _atomic_json(args.output / "records.json", {"records": records})

    result = {
        "status": "COMPLETE",
        "method": "Qwen Frame Prompt mask -> official SAM3.1 tracker memory",
        "ground_truth_used_for_initialization_or_selection": False,
        "frame_prompt": _aggregate(records, "frame"),
        "qwen_seeded_tracker": _aggregate(records, "tracker"),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    _atomic_json(args.output / "result.json", result)
    status.update({"status": "complete", "completed_samples": len(dataset)})
    _atomic_json(args.output / "STATUS.json", status)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--qwen-pixels", type=int, default=100352)
    parser.add_argument("--sam-visual-chunk-size", type=int, default=1)
    parser.add_argument("--sam-decode-chunk-size", type=int, default=1)
    parser.add_argument(
        "--qwen-checkpoint", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/models/SAM3.1-official-mirror/sam3.1_multiplex.pt"),
    )
    parser.add_argument(
        "--sam-repo", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/external/sam3"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run(arguments)
    print(json.dumps(summary["qwen_seeded_tracker"], indent=2))

"""Evaluate the official SAM3.1 detector-to-tracker video path.

This is a read-only foundation audit: the referring expression is supplied to
the official detector, its object identity is carried by the official tracker,
and no ground-truth mask is used to initialize or select a track.  The result
decides whether tracker memory is a sufficiently strong base for the next FTG
interface before any additional Qwen training is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from projects.evoseg.ftg.metrics import evaluate_masks
from projects.evoseg.ftg.public_video_data import PublicVideoPilotDataset
from projects.evoseg.qwen_process_seg.sam31_executor import FrozenSAM31Executor


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


def _materialize_sampled_video(image_paths: list[str], directory: Path) -> None:
    """Expose only the sampled frames to the official ordered-folder loader."""
    for index, source in enumerate(image_paths):
        destination = directory / f"{index:06d}.jpg"
        destination.symlink_to(Path(source).resolve())


def _persistent_track_prediction(
    outputs: dict[int, dict],
    object_scores: dict[int, float],
    frame_count: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int | None]:
    """Select one official track once, then retain that identity in every frame."""
    # The official tracker retains scores for tracks later suppressed or removed.
    # Restrict selection to tracks exposed by its own post-processing, without
    # consulting ground truth.
    observed_ids = {
        int(object_id)
        for output in outputs.values()
        for object_id in np.asarray(output["out_obj_ids"]).tolist()
    }
    candidates = observed_ids.intersection(object_scores)
    if not candidates:
        return np.zeros((frame_count, *shape), dtype=bool), None
    object_id = max(
        candidates, key=lambda key: (float(object_scores[key]), -int(key))
    )
    prediction = np.zeros((frame_count, *shape), dtype=bool)
    for frame_index, output in outputs.items():
        ids = np.asarray(output["out_obj_ids"]).tolist()
        if object_id in ids:
            prediction[frame_index] = np.asarray(
                output["out_binary_masks"][ids.index(object_id)], dtype=bool
            )
    return prediction, int(object_id)


def _union_prediction(
    outputs: dict[int, dict], frame_count: int, shape: tuple[int, int]
) -> np.ndarray:
    prediction = np.zeros((frame_count, *shape), dtype=bool)
    for frame_index, output in outputs.items():
        masks = np.asarray(output["out_binary_masks"], dtype=bool)
        if masks.size:
            prediction[frame_index] = masks.any(axis=0)
    return prediction


def _aggregate(records: list[dict], prefix: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        groups[record["dataset"]].append(record)
        groups[f'{record["dataset"]}/{record["expression_type"]}'].append(record)
    metrics = ("j", "f", "jf", "present_jf", "false_accept", "false_reject")
    return {
        group: {
            "count": len(values),
            **{
                metric: float(np.mean([value[f"{prefix}_{metric}"] for value in values]))
                for metric in metrics
            },
        }
        for group, values in sorted(groups.items())
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)

    manifest = json.loads(args.manifest.read_text())
    dataset = PublicVideoPilotDataset(manifest, args.partition)
    if args.max_records is not None:
        dataset.records = dataset.records[: args.max_records]
    if not dataset:
        raise RuntimeError("selected manifest partition is empty")

    executor = FrozenSAM31Executor(args.sam_checkpoint, args.sam_repo)
    model = executor.assembled
    records = []
    started = time.time()
    status = {
        "schema_version": 1,
        "stage": "sam31_native_video_baseline",
        "status": "running",
        "pid": os.getpid(),
        "device": args.device,
        "partition": args.partition,
        "sample_count": len(dataset),
        "completed_samples": 0,
        "manifest_sha256": _sha256(args.manifest),
        "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
    }
    _atomic_json(args.output / "STATUS.json", status)

    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["masks"]
        shape = tuple(target.shape[-2:])
        with tempfile.TemporaryDirectory(prefix="sam31-ftg-video-", dir="/tmp") as temporary:
            directory = Path(temporary)
            _materialize_sampled_video(sample["image_paths"], directory)
            state = model.init_state(
                resource_path=str(directory),
                offload_video_to_cpu=False,
                async_loading_frames=False,
                use_cv2=False,
            )
            model.add_prompt(
                state,
                frame_idx=0,
                text_str=sample["expression"],
            )
            outputs = {
                int(frame_index): output
                for frame_index, output in model.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(sample["image_paths"]),
                    reverse=False,
                    is_last_batch=True,
                )
            }
            object_scores = {
                int(key): float(value)
                for key, value in state["tracker_metadata"].get(
                    "obj_id_to_score", {}
                ).items()
            }

        top1, chosen_track = _persistent_track_prediction(
            outputs, object_scores, len(sample["image_paths"]), shape
        )
        union = _union_prediction(outputs, len(sample["image_paths"]), shape)
        top1_metrics = evaluate_masks(target, top1)
        union_metrics = evaluate_masks(target, union)
        record = {
            "dataset": sample["dataset"],
            "video_id": sample["video_id"],
            "expression_id": sample["expression_id"],
            "expression_type": sample["expression_type"],
            "expression": sample["expression"],
            "chosen_track": chosen_track,
            "detected_track_count": len(object_scores),
            "object_scores": object_scores,
            **{f"top1_{key}": value for key, value in top1_metrics.items()},
            **{f"union_{key}": value for key, value in union_metrics.items()},
        }
        records.append(record)
        status.update(
            {
                "completed_samples": index + 1,
                "elapsed_seconds": time.time() - started,
                "eta_seconds": (time.time() - started)
                / (index + 1)
                * (len(dataset) - index - 1),
                "last_record": {
                    "dataset": sample["dataset"],
                    "video_id": sample["video_id"],
                    "top1_jf": record["top1_jf"],
                    "union_jf": record["union_jf"],
                },
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
        )
        _atomic_json(args.output / "STATUS.json", status)
        _atomic_json(args.output / "records.json", {"records": records})

    result = {
        "status": "COMPLETE",
        "foundation": "official SAM3.1 detector-to-tracker",
        "ground_truth_used_for_initialization_or_selection": False,
        "top1_track": _aggregate(records, "top1"),
        "union_all_tracks": _aggregate(records, "union"),
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-repo", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments)["top1_track"], indent=2))

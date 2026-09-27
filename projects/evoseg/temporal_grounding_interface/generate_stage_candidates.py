"""Generate current-stage SAM 3.1 candidate objects without reading GT.

Each official concept prompt is evaluated only at the current anchor frame.
There is no full-track propagation in candidate generation and no candidate is
selected here.  The public semantic prompt resets its own candidate session as
documented by the dynamic-interface audit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    SpacyConceptParser,
    audit_loaded_checkpoint,
    sha256,
    start_multiplex_session,
)
from projects.evoseg.temporal_grounding_interface.protocol import (
    stable_identity,
    stage_end_positions,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def encode_rle(mask: np.ndarray) -> dict:
    value = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    value["counts"] = value["counts"].decode("ascii")
    return value


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            value["identity"]
            for line in handle
            if line.strip()
            for value in [json.loads(line)]
            if value.get("status") == "success"
        }


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def output_candidates(outputs: dict) -> list[dict]:
    ids = [int(value) for value in np.asarray(outputs["out_obj_ids"]).reshape(-1)]
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    probabilities = outputs.get("out_probs")
    probabilities = (
        [None] * len(ids)
        if probabilities is None
        else [float(value) for value in np.asarray(probabilities).reshape(-1)]
    )
    if len(ids) != len(masks) or len(ids) != len(probabilities):
        raise RuntimeError("SAM3.1 output ID/mask/confidence lengths differ")
    return [
        {
            "official_index": index,
            "object_id": object_id,
            "confidence": probabilities[index],
            "mask": encode_rle(masks[index]),
        }
        for index, object_id in enumerate(ids)
    ]


def run(args) -> int:
    sam3_repo = Path(args.sam3_repo).resolve()
    sys.path.insert(0, str(sam3_repo))
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    source = Path(inspect.getfile(build_sam3_multiplex_video_predictor)).resolve()
    if sam3_repo not in source.parents:
        raise RuntimeError(f"SAM3 imported outside official checkout: {source}")
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint_hash}")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    objects = manifest["objects"][: args.max_objects] if args.max_objects else manifest["objects"]
    objects = [
        item for index, item in enumerate(objects) if index % args.num_shards == args.shard_index
    ]
    run_dir = Path(args.run_dir).resolve()
    (run_dir / "stage_candidates").mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "stage_candidate_records.jsonl"
    done = completed(records_path)
    parser = SpacyConceptParser(args.spacy_model)

    torch.cuda.set_device(args.device)
    load_started = time.perf_counter()
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
        run_dir / "stage_candidate_config.json",
        {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "evoseg_commit": git_output(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"),
            "official_repo": "https://github.com/facebookresearch/sam3",
            "official_repo_path": str(sam3_repo),
            "official_repo_commit": git_output(sam3_repo, "rev-parse", "HEAD"),
            "runtime_builder_source": str(source),
            "runtime_builder_sha256": sha256(source),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "weight_audit": weight_audit,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "model_load_seconds_synchronized": time.perf_counter() - load_started,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device),
                "gpu_index": args.device,
            },
            "protocol": {
                "candidate_generation_reads_gt": False,
                "candidate_generation_propagates_tracks": False,
                "prompt": "deterministic official-expression noun phrase",
                "anchors": "nested canonical K=8 stage endpoints",
                "selected_objects": len(objects),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            "concept_parser": {
                "name": "spacy_subject_noun_chunk_v1",
                "spacy_version": parser.spacy_version,
                "model": parser.model_name,
            },
        },
    )

    planned = sum(len(item["expressions"]) for item in objects)
    attempted = failed = 0
    started = time.monotonic()
    image_root = Path(manifest["dataset"]["image_root"])
    for item in objects:
        item_identities = {
            stable_identity(
                item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]
            )
            for expression in item["expressions"]
        }
        # A resumed run must not pay the (potentially very large) cost of
        # constructing an official SAM video session for an object whose
        # expressions are already durable on disk.
        if item_identities.issubset(done):
            continue
        video_path = image_root / item["video_id"]
        session_id = None
        try:
            session, compatibility = start_multiplex_session(predictor, str(video_path))
            session_id = session["session_id"]
            endpoints = stage_end_positions(item["frame_count"], 8)
            for expression in item["expressions"]:
                identity = stable_identity(
                    item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]
                )
                if identity in done:
                    continue
                parse = parser(expression["text"])
                record = {
                    "identity": identity,
                    "dataset": item["dataset"],
                    "video_id": item["video_id"],
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "prompt": parse["concept"],
                    "concept_parse": parse,
                    "stage_end_indices": endpoints,
                    "gt_read_during_generation": False,
                    "full_track_propagation_during_generation": False,
                    "started_at": utc_now(),
                }
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    query_started = time.perf_counter()
                    stages = []
                    for stage_index, endpoint in enumerate(endpoints):
                        torch.cuda.synchronize()
                        stage_started = time.perf_counter()
                        response = predictor.add_prompt(
                            session_id=session_id,
                            frame_idx=endpoint,
                            text=parse["concept"],
                        )
                        torch.cuda.synchronize()
                        stages.append(
                            {
                                "stage_index": stage_index,
                                "anchor_frame_index": int(response["frame_index"]),
                                "anchor_frame_name": item["frame_names"][endpoint],
                                "latency_seconds_synchronized": time.perf_counter()
                                - stage_started,
                                "candidates": output_candidates(response["outputs"]),
                            }
                        )
                    torch.cuda.synchronize()
                    relative = Path("stage_candidates") / f"{identity.replace('/', '__')}.json"
                    atomic_json(
                        run_dir / relative,
                        {
                            "identity": identity,
                            "prompt": parse["concept"],
                            "stages": stages,
                        },
                    )
                    record.update(
                        {
                            "status": "success",
                            "completed_at": utc_now(),
                            "candidate_path": str(relative),
                            "candidate_counts": [len(stage["candidates"]) for stage in stages],
                            "latency_seconds_synchronized": time.perf_counter() - query_started,
                            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "session_api_compat": compatibility,
                        }
                    )
                    done.add(identity)
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
                    run_dir / "CANDIDATE_STATUS.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "planned": planned,
                        "completed": len(done),
                        "attempted_this_process": attempted,
                        "failed_this_process": failed,
                        "current_identity": identity,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / attempted * max(planned - len(done), 0),
                        "updated_at": utc_now(),
                    },
                )
        finally:
            if session_id is not None:
                predictor.close_session(session_id)
    atomic_json(
        run_dir / "CANDIDATE_STATUS.json",
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
    predictor.shutdown()
    return 0 if failed == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

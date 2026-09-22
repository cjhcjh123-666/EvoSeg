"""SAM 3.1 candidate-bank generation and GT-separated evaluation.

Generation intentionally has no ground-truth argument.  Evaluation is a
separate command that reads the saved candidate RLEs only after generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(directory: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True
        ).strip()
    except Exception:
        return None


def candidate_key(item: dict, expression: dict, prompt_method: str) -> str:
    return "/".join(
        [
            item["dataset"],
            item["video_id"],
            item["object_id"],
            expression["expression_id"],
            prompt_method,
        ]
    )


def select_pilot_objects(
    objects: list[dict],
    limit: int | None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    # The upstream manifest is already prediction-independent, seed-42 order.
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"invalid shard {shard_index} of {num_shards}; expected 0 <= index < count"
        )
    selected = objects if limit is None else objects[:limit]
    return [item for index, item in enumerate(selected) if index % num_shards == shard_index]


class SpacyConceptParser:
    """Deterministically select the grammatical subject noun phrase."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        import spacy

        self.spacy_version = spacy.__version__
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    @staticmethod
    def _clean_span(span) -> str:
        tokens = [
            token.text
            for token in span
            if token.dep_ not in {"det", "predet"} and not token.is_punct
        ]
        return " ".join(tokens).strip()

    def __call__(self, expression: str) -> dict:
        document = self.nlp(expression)
        noun_chunks = list(document.noun_chunks)
        subject_chunks = [
            chunk
            for chunk in noun_chunks
            if chunk.root.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
        ]
        candidates = subject_chunks or noun_chunks
        if not candidates:
            raise ValueError(f"no noun phrase found: {expression!r}")
        span = min(candidates, key=lambda value: value.start)
        concept = self._clean_span(span)
        if not concept:
            raise ValueError(f"empty concept after determiner removal: {expression!r}")
        return {
            "concept": concept,
            "selected_span": span.text,
            "root": span.root.text,
            "root_pos": span.root.pos_,
            "root_dependency": span.root.dep_,
            "parser": "spacy_subject_noun_chunk_v1",
            "spacy_version": self.spacy_version,
            "spacy_model": self.model_name,
        }


def encode_rle(mask: np.ndarray) -> dict:
    value = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    value["counts"] = value["counts"].decode("ascii")
    return value


def decode_rle(value: dict) -> np.ndarray:
    rle = dict(value)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def save_candidate_tracks(
    path: Path,
    evaluation_indices: list[int],
    output_by_frame: dict[int, dict],
    image_shape: tuple[int, int],
) -> dict:
    track_ids = sorted(
        {
            int(track_id)
            for output in output_by_frame.values()
            for track_id in output["out_obj_ids"]
        }
    )
    height, width = image_shape
    tracks = []
    for track_id in track_ids:
        frames = []
        confidences = []
        for frame_index in evaluation_indices:
            output = output_by_frame.get(frame_index)
            mask = np.zeros((height, width), dtype=bool)
            if output is not None:
                ids = [int(value) for value in output["out_obj_ids"]]
                if track_id in ids:
                    position = ids.index(track_id)
                    mask = np.asarray(output["out_binary_masks"][position], dtype=bool)
                    confidences.append(float(output["out_probs"][position]))
            frames.append(encode_rle(mask))
        tracks.append(
            {
                "track_id": track_id,
                "mean_confidence": (
                    float(np.mean(confidences)) if confidences else None
                ),
                "confidence_observations": len(confidences),
                "frames": frames,
            }
        )
    value = {
        "evaluation_frame_indices": evaluation_indices,
        "height": height,
        "width": width,
        "tracks": tracks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return {"candidate_count": len(tracks), "candidate_track_ids": track_ids}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_keys(path: Path) -> set[str]:
    return {
        record["key"]
        for record in load_records(path)
        if record.get("status") == "success"
    }


def expression_key(item: dict, expression: dict) -> str:
    return "/".join(
        [
            item["dataset"], item["video_id"], str(item["object_id"]),
            str(expression["expression_id"]),
        ]
    )


def load_qwen_concepts(path_value: str | None) -> tuple[dict[str, dict], Path | None]:
    if not path_value:
        return {}, None
    path = Path(path_value).resolve()
    records = load_records(path)
    successful = [record for record in records if record.get("status") == "success"]
    concepts = {record["key"]: record for record in successful}
    if len(concepts) != len(successful):
        raise RuntimeError(f"duplicate successful Qwen concept key in {path}")
    return concepts, path


def run_generation(args) -> int:
    # Import from the exact official checkout only at runtime.  The recorded
    # source path and commit make accidental use of a vendored fork visible.
    sam3_repo = Path(args.sam3_repo).resolve()
    sys.path.insert(0, str(sam3_repo))
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    module_path = Path(inspect.getfile(build_sam3_multiplex_video_predictor)).resolve()
    if sam3_repo not in module_path.parents:
        raise RuntimeError(f"SAM3 imported from unexpected source: {module_path}")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"official SAM3.1 checkpoint missing: {checkpoint}")
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "SAM3.1 checkpoint SHA256 mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    objects = select_pilot_objects(
        manifest["objects"],
        args.max_objects,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    run_dir = Path(args.run_dir).resolve()
    records_path = run_dir / "candidate_records.jsonl"
    done = completed_keys(records_path)
    parser = SpacyConceptParser(args.spacy_model)
    qwen_concepts, qwen_concepts_path = load_qwen_concepts(args.qwen_concepts)
    if qwen_concepts_path is not None:
        missing = [
            expression_key(item, expression)
            for item in objects
            for expression in item["expressions"]
            if expression_key(item, expression) not in qwen_concepts
        ]
        if missing:
            raise RuntimeError(
                f"Qwen concept map misses {len(missing)} selected expressions; first={missing[0]}"
            )

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
    config = {
        "created_at": utc_now(),
        "official_repo": "https://github.com/facebookresearch/sam3",
        "official_repo_path": str(sam3_repo),
        "official_repo_commit": git_output(sam3_repo, "rev-parse", "HEAD"),
        "runtime_builder_source": str(module_path),
        "runtime_builder_source_sha256": sha256(module_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_expected_sha256": args.expected_checkpoint_sha256,
        "checkpoint_source": args.checkpoint_source,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_load_seconds_synchronized": time.perf_counter() - load_started,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_index": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
        },
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "prompt_methods": ["raw_expression", "concept"]
            + (["qwen_concept"] if qwen_concepts_path is not None else []),
            "prompt_frame_index": 0,
            "candidate_generation_reads_gt": False,
            "max_num_objects": args.max_num_objects,
            "multiplex_count": args.multiplex_count,
            "use_fa3": args.use_fa3,
            "compile": args.compile,
            "selected_objects": len(objects),
            "selection": "first objects in prediction-independent seed-42 manifest order, then object-level modulo shard",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "qwen_concepts": str(qwen_concepts_path) if qwen_concepts_path else None,
            "qwen_concepts_sha256": sha256(qwen_concepts_path)
            if qwen_concepts_path
            else None,
        },
        "concept_parser": {
            "implementation": "spacy_subject_noun_chunk_v1",
            "spacy_version": parser.spacy_version,
            "model": parser.model_name,
        },
    }
    atomic_json(run_dir / "run_config.json", config)

    prompt_method_count = 3 if qwen_concepts_path is not None else 2
    planned = sum(len(item["expressions"]) * prompt_method_count for item in objects)
    attempted = 0
    failed = 0
    started = time.monotonic()
    atomic_json(
        run_dir / "STATUS.json",
        {
            "state": "running",
            "pid": os.getpid(),
            "planned": planned,
            "completed": len(done),
            "failed_this_process": 0,
            "updated_at": utc_now(),
        },
    )
    for item in objects:
        video_path = Path(manifest["dataset"]["image_root"]) / item["video_id"]
        with Image.open(video_path / f"{item['frame_names'][0]}.jpg") as image:
            image_shape = (image.height, image.width)
        for expression in item["expressions"]:
            concept = parser(expression["text"])
            prompts = [
                ("raw_expression", expression["text"]),
                ("concept", concept["concept"]),
            ]
            if qwen_concepts_path is not None:
                prompts.append(
                    (
                        "qwen_concept",
                        qwen_concepts[expression_key(item, expression)]["concept"],
                    )
                )
            for prompt_method, prompt in prompts:
                key = candidate_key(item, expression, prompt_method)
                if key in done:
                    continue
                base = {
                    "key": key,
                    "dataset": item["dataset"],
                    "video_id": item["video_id"],
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "prompt_method": prompt_method,
                    "prompt": prompt,
                    "concept_parse": concept if prompt_method == "concept" else None,
                    "qwen_concept_source_key": expression_key(item, expression)
                    if prompt_method == "qwen_concept"
                    else None,
                    "prompt_frame_index": 0,
                    "evaluation_frame_indices": item["evaluation_frame_indices"],
                    "gt_read_during_generation": False,
                    "started_at": utc_now(),
                }
                session_id = None
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    query_started = time.perf_counter()
                    response = predictor.handle_request(
                        {"type": "start_session", "resource_path": str(video_path)}
                    )
                    session_id = response["session_id"]
                    initial = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": prompt,
                        }
                    )
                    output_by_frame = {int(initial["frame_index"]): initial["outputs"]}
                    for response in predictor.handle_stream_request(
                        {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": "forward",
                        }
                    ):
                        output_by_frame[int(response["frame_index"])] = response["outputs"]
                    torch.cuda.synchronize()
                    relative_path = (
                        Path("candidate_tracks") / f"{key.replace('/', '__')}.json"
                    )
                    candidate_info = save_candidate_tracks(
                        run_dir / relative_path,
                        item["evaluation_frame_indices"],
                        output_by_frame,
                        image_shape,
                    )
                    base.update(
                        {
                            "status": "success",
                            "completed_at": utc_now(),
                            "candidate_tracks_path": str(relative_path),
                            "latency_seconds_synchronized": time.perf_counter()
                            - query_started,
                            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            **candidate_info,
                        }
                    )
                    done.add(key)
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    failed += 1
                    base.update(
                        {
                            "status": "failed_oom",
                            "completed_at": utc_now(),
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                except Exception as error:
                    failed += 1
                    base.update(
                        {
                            "status": "failed",
                            "completed_at": utc_now(),
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                finally:
                    if session_id is not None:
                        predictor.handle_request(
                            {"type": "close_session", "session_id": session_id}
                        )
                append_jsonl(records_path, base)
                attempted += 1
                elapsed = time.monotonic() - started
                remaining = max(planned - len(done), 0)
                atomic_json(
                    run_dir / "STATUS.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "planned": planned,
                        "completed": len(done),
                        "attempted_this_process": attempted,
                        "failed_this_process": failed,
                        "elapsed_seconds_this_process": elapsed,
                        "estimated_remaining_seconds": (
                            elapsed / attempted * remaining if attempted else None
                        ),
                        "current_key": key,
                        "updated_at": utc_now(),
                    },
                )
    atomic_json(
        run_dir / "STATUS.json",
        {
            "state": "complete" if failed == 0 else "complete_with_failures",
            "pid": os.getpid(),
            "planned": planned,
            "completed": len(done),
            "attempted_this_process": attempted,
            "failed_this_process": failed,
            "elapsed_seconds_this_process": time.monotonic() - started,
            "estimated_remaining_seconds": 0.0,
            "current_key": None,
            "updated_at": utc_now(),
        },
    )
    return 0 if failed == 0 else 2


def load_ground_truth(item: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for path_value in item["evaluation_mask_paths"]:
        path = Path(path_value)
        if path.is_file():
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.shape != shape:
                raise ValueError(f"GT shape mismatch: {mask.shape} != {shape}: {path}")
        else:
            mask = np.zeros(shape, dtype=bool)
        masks.append(mask)
    return np.stack(masks)


def evaluate_record(record: dict, run_dir: Path, item: dict) -> dict:
    value = json.loads((run_dir / record["candidate_tracks_path"]).read_text())
    if value["evaluation_frame_indices"] != item["evaluation_frame_indices"]:
        raise AssertionError("candidate/manifest evaluation frames differ")
    shape = (value["height"], value["width"])
    ground_truth = load_ground_truth(item, shape)
    candidates = []
    for track in value["tracks"]:
        masks = np.stack([decode_rle(frame) for frame in track["frames"]])
        j_values = db_eval_iou(ground_truth, masks)
        f_values = db_eval_boundary(ground_truth, masks)
        j_score = float(np.mean(j_values))
        f_score = float(np.mean(f_values))
        candidates.append(
            {
                "track_id": track["track_id"],
                "confidence": track["mean_confidence"],
                "J": j_score,
                "F": f_score,
                "J_and_F": (j_score + f_score) / 2,
            }
        )
    best = max(candidates, key=lambda candidate: candidate["J_and_F"], default=None)
    return {
        "key": record["key"],
        "dataset": record["dataset"],
        "video_id": record["video_id"],
        "object_id": record["object_id"],
        "expression_id": record["expression_id"],
        "description_type": record["description_type"],
        "expression": record["expression"],
        "prompt_method": record["prompt_method"],
        "prompt": record["prompt"],
        "candidate_count": len(candidates),
        "oracle_track_id": best["track_id"] if best else None,
        "oracle_J": best["J"] if best else 0.0,
        "oracle_F": best["F"] if best else 0.0,
        "oracle_J_and_F": best["J_and_F"] if best else 0.0,
        "recall_at_0_3": int(best is not None and best["J_and_F"] >= 0.3),
        "recall_at_0_5": int(best is not None and best["J_and_F"] >= 0.5),
        "recall_at_0_7": int(best is not None and best["J_and_F"] >= 0.7),
        "candidate_metrics": json.dumps(candidates, separators=(",", ":")),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    object_lookup = {
        (item["dataset"], item["video_id"], item["object_id"]): item
        for item in manifest["objects"]
    }
    records = [
        record
        for record in load_records(run_dir / "candidate_records.jsonl")
        if record.get("status") == "success"
    ]
    rows = [
        evaluate_record(
            record,
            run_dir,
            object_lookup[(record["dataset"], record["video_id"], record["object_id"])],
        )
        for record in records
    ]
    write_csv(run_dir / "sam31_candidate_metrics.csv", rows)
    summary = []
    for prompt_method in ("raw_expression", "concept"):
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in rows
                if row["prompt_method"] == prompt_method
                and row["description_type"] == description_type
            ]
            summary.append(
                {
                    "prompt_method": prompt_method,
                    "description_type": description_type,
                    "expressions": len(selected),
                    "objects": len(
                        {(row["video_id"], row["object_id"]) for row in selected}
                    ),
                    "mean_oracle_J": float(np.mean([row["oracle_J"] for row in selected]))
                    if selected
                    else None,
                    "mean_oracle_F": float(np.mean([row["oracle_F"] for row in selected]))
                    if selected
                    else None,
                    "mean_oracle_J_and_F": float(
                        np.mean([row["oracle_J_and_F"] for row in selected])
                    )
                    if selected
                    else None,
                    "recall_at_0_3": float(
                        np.mean([row["recall_at_0_3"] for row in selected])
                    )
                    if selected
                    else None,
                    "recall_at_0_5": float(
                        np.mean([row["recall_at_0_5"] for row in selected])
                    )
                    if selected
                    else None,
                    "recall_at_0_7": float(
                        np.mean([row["recall_at_0_7"] for row in selected])
                    )
                    if selected
                    else None,
                    "mean_candidate_count": float(
                        np.mean([row["candidate_count"] for row in selected])
                    )
                    if selected
                    else None,
                }
            )
    write_csv(run_dir / "sam31_candidate_summary.csv", summary)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generation = subparsers.add_parser("generate")
    generation.add_argument("--manifest", required=True)
    generation.add_argument("--run-dir", required=True)
    generation.add_argument("--checkpoint", required=True)
    generation.add_argument("--expected-checkpoint-sha256", required=True)
    generation.add_argument("--checkpoint-source", required=True)
    generation.add_argument("--sam3-repo", required=True)
    generation.add_argument("--device", type=int, required=True)
    generation.add_argument("--max-objects", type=int)
    generation.add_argument("--shard-index", type=int, default=0)
    generation.add_argument("--num-shards", type=int, default=1)
    generation.add_argument("--max-num-objects", type=int, default=16)
    generation.add_argument("--multiplex-count", type=int, default=16)
    generation.add_argument("--spacy-model", default="en_core_web_sm")
    generation.add_argument("--qwen-concepts")
    generation.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=False)
    generation.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--run-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "generate":
        raise SystemExit(run_generation(arguments))
    raise SystemExit(run_evaluation(arguments))

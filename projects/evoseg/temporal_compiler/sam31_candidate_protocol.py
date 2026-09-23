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
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_multiplex_session(predictor, resource_path: str) -> tuple[dict, dict]:
    """Start an official SAM3.1 session across the 3.0/3.1 API mismatch.

    The current official ``Sam3BasePredictor.start_session`` always forwards
    ``offload_state_to_cpu``. The official 3.1 Multiplex model removed that
    argument from ``init_state``. Keep the compatibility shim in this adapter
    (not in the vendored Meta checkout), and omit only arguments absent from
    the loaded model's inspected signature.
    """
    init_signature = inspect.signature(predictor.model.init_state)
    supported = set(init_signature.parameters)
    requested = {
        "resource_path": resource_path,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    }
    if hasattr(predictor, "async_loading_frames"):
        requested["async_loading_frames"] = predictor.async_loading_frames
    if hasattr(predictor, "video_loader_type"):
        requested["video_loader_type"] = predictor.video_loader_type
    forwarded = {key: value for key, value in requested.items() if key in supported}
    omitted = sorted(set(requested) - set(forwarded))

    inference_state = predictor.model.init_state(**forwarded)
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    return {"session_id": session_id}, {
        "model_init_state_signature": str(init_signature),
        "forwarded_arguments": sorted(forwarded),
        "omitted_unsupported_arguments": omitted,
        "official_base_predictor_mismatch": "offload_state_to_cpu" in omitted,
    }


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


def audit_loaded_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(payload)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    allowed_missing = [
        key
        for key in missing
        if key.endswith(".attn.freqs_cis_real")
        or key.endswith(".attn.freqs_cis_imag")
    ]
    disallowed_missing = sorted(set(missing) - set(allowed_missing))
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "final SAM3.1 multiplex checkpoint/model key mismatch: "
            f"missing={disallowed_missing[:10]} ({len(disallowed_missing)}), "
            f"unexpected={unexpected[:10]} ({len(unexpected)})"
        )
    ordered = sorted(model_keys & checkpoint_keys)
    sample_positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
    verified = []
    for position in sample_positions:
        key = ordered[position]
        model_value = model_state[key].detach().reshape(-1)
        checkpoint_value = payload[key].detach().reshape(-1)
        if model_value.shape != checkpoint_value.shape:
            raise RuntimeError(f"checkpoint tensor shape mismatch for {key}")
        sample_length = min(16, model_value.numel())
        if not torch.equal(
            model_value[:sample_length].cpu(), checkpoint_value[:sample_length]
        ):
            raise RuntimeError(f"checkpoint tensor value mismatch for {key}")
        verified.append(key)
    del payload
    return {
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "missing_keys": allowed_missing,
        "missing_keys_reason": (
            "deterministically rebuilt real/imaginary RoPE buffers under "
            "official use_rope_real=True"
            if allowed_missing
            else None
        ),
        "disallowed_missing_keys": [],
        "unexpected_keys": [],
        "sample_value_keys_verified": verified,
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
    final_weight_audit = audit_loaded_checkpoint(predictor.model, checkpoint)
    evoseg_repo = Path(__file__).resolve().parents[3]
    config = {
        "created_at": utc_now(),
        "evoseg_repo_path": str(evoseg_repo),
        "evoseg_repo_commit": git_output(evoseg_repo, "rev-parse", "HEAD"),
        "adapter_source": str(Path(__file__).resolve()),
        "adapter_source_sha256": sha256(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
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
        "final_weight_audit": final_weight_audit,
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
            "max_objects": args.max_objects,
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
                    response, session_api_compat = start_multiplex_session(
                        predictor, str(video_path)
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
                            "session_api_compat": session_api_compat,
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
    present_flags = item.get("evaluation_mask_present")
    if present_flags is None:
        raise KeyError(
            f"manifest object {item['video_id']}/{item['object_id']} lacks "
            "evaluation_mask_present"
        )
    if len(present_flags) != len(item["evaluation_mask_paths"]):
        raise ValueError(
            f"evaluation mask presence/path length mismatch for "
            f"{item['video_id']}/{item['object_id']}"
        )
    masks = []
    for path_value, expected_present in zip(
        item["evaluation_mask_paths"], present_flags
    ):
        path = Path(path_value)
        file_present = path.is_file()
        if bool(expected_present) != file_present:
            raise FileNotFoundError(
                f"evaluation GT availability differs from manifest for "
                f"{item['video_id']}/{item['object_id']}: expected_present="
                f"{bool(expected_present)}, file_present={file_present}, path={path}"
            )
        if file_present:
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


def evaluate_record_task(task: tuple[dict, dict]) -> dict:
    record, item = task
    return evaluate_record(record, Path(record["_source_run_dir"]), item)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_candidate_rows(
    rows: list[dict], prompt_methods_override: list[str] | None = None
) -> list[dict]:
    summary = []
    prompt_methods = (
        sorted(prompt_methods_override)
        if prompt_methods_override is not None
        else sorted({row["prompt_method"] for row in rows})
    )
    score_fields = (
        "oracle_J",
        "oracle_F",
        "oracle_J_and_F",
        "recall_at_0_3",
        "recall_at_0_5",
        "recall_at_0_7",
        "candidate_count",
    )
    for prompt_method in prompt_methods:
        for description_type in ("static", "dynamic", "hybrid"):
            selected = [
                row
                for row in rows
                if row["prompt_method"] == prompt_method
                and row["description_type"] == description_type
            ]
            by_object = defaultdict(list)
            for row in selected:
                by_object[(row["dataset"], row["video_id"], row["object_id"])].append(
                    row
                )
            for aggregation in ("expression_weighted", "object_weighted"):
                if aggregation == "expression_weighted":
                    values = selected
                else:
                    values = [
                        {
                            field: float(np.mean([row[field] for row in object_rows]))
                            for field in score_fields
                        }
                        for object_rows in by_object.values()
                    ]
                counts = [value["candidate_count"] for value in values]
                summary.append(
                    {
                        "prompt_method": prompt_method,
                        "description_type": description_type,
                        "aggregation": aggregation,
                        "expressions": len(selected),
                        "objects": len(by_object),
                        "mean_oracle_J": float(np.mean([v["oracle_J"] for v in values]))
                        if values
                        else None,
                        "mean_oracle_F": float(np.mean([v["oracle_F"] for v in values]))
                        if values
                        else None,
                        "mean_oracle_J_and_F": float(
                            np.mean([v["oracle_J_and_F"] for v in values])
                        )
                        if values
                        else None,
                        "recall_at_0_3": float(
                            np.mean([v["recall_at_0_3"] for v in values])
                        )
                        if values
                        else None,
                        "recall_at_0_5": float(
                            np.mean([v["recall_at_0_5"] for v in values])
                        )
                        if values
                        else None,
                        "recall_at_0_7": float(
                            np.mean([v["recall_at_0_7"] for v in values])
                        )
                        if values
                        else None,
                        "mean_candidate_count": float(np.mean(counts)) if counts else None,
                        "median_candidate_count": float(np.median(counts))
                        if counts
                        else None,
                        "p95_candidate_count": float(np.percentile(counts, 95))
                        if counts
                        else None,
                        "min_candidate_count": float(np.min(counts)) if counts else None,
                        "max_candidate_count": float(np.max(counts)) if counts else None,
                    }
                )
    return summary


def build_candidate_completeness(
    manifest: dict, expected_keys: set[str], successful_keys: set[str]
) -> list[dict]:
    """Audit expression completeness for every prompt/type/object cell."""

    prompt_methods = sorted({key.rsplit("/", 1)[-1] for key in expected_keys})
    expected_by_cell = defaultdict(list)
    for item in manifest["objects"]:
        object_key = (item["dataset"], item["video_id"], str(item["object_id"]))
        for expression in item["expressions"]:
            for prompt_method in prompt_methods:
                key = candidate_key(item, expression, prompt_method)
                if key in expected_keys:
                    expected_by_cell[
                        (prompt_method, expression["type"], *object_key)
                    ].append((str(expression["expression_id"]), key))
    rows = []
    for (prompt_method, description_type, dataset, video_id, object_id), values in sorted(
        expected_by_cell.items()
    ):
        succeeded = [value for value in values if value[1] in successful_keys]
        missing = [value for value in values if value[1] not in successful_keys]
        rows.append(
            {
                "prompt_method": prompt_method,
                "description_type": description_type,
                "dataset": dataset,
                "video_id": video_id,
                "object_id": object_id,
                "expected_expressions": len(values),
                "successful_expressions": len(succeeded),
                "missing_expressions": len(missing),
                "complete": int(not missing),
                "partial": int(bool(succeeded) and bool(missing)),
                "zero_success": int(not succeeded),
                "missing_expression_ids": ";".join(value[0] for value in missing),
                "missing_keys": ";".join(value[1] for value in missing),
            }
        )
    return rows


def aggregate_complete_object_candidate_rows(
    rows: list[dict], completeness: list[dict]
) -> list[dict]:
    """Compute coverage only where all official expressions in a cell succeeded."""

    complete_cells = {
        (
            row["prompt_method"],
            row["description_type"],
            row["dataset"],
            row["video_id"],
            str(row["object_id"]),
        )
        for row in completeness
        if row["complete"]
    }
    selected = [
        row
        for row in rows
        if (
            row["prompt_method"],
            row["description_type"],
            row["dataset"],
            row["video_id"],
            str(row["object_id"]),
        )
        in complete_cells
    ]
    prompt_methods = sorted({row["prompt_method"] for row in completeness})
    summaries = aggregate_candidate_rows(selected, prompt_methods)
    output = []
    for summary in summaries:
        if summary["aggregation"] != "object_weighted":
            continue
        value = dict(summary)
        value["aggregation"] = "complete_object_weighted"
        output.append(value)
    return output


def summarize_generation_attempts(
    records: list[dict], expected_successful_records: int | None
) -> dict:
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") != "success"]
    attempted_keys = [record["key"] for record in records]
    unique_attempted_keys = set(attempted_keys)
    unique_successful_keys = {record["key"] for record in successful}
    missing_successful = (
        max(expected_successful_records - len(unique_successful_keys), 0)
        if expected_successful_records is not None
        else None
    )
    return {
        # Preserve old field names while explicitly distinguishing unique
        # records from attempts retained across retries.
        "generation_records": len(records),
        "generation_attempt_records": len(records),
        "successful_generation_records": len(successful),
        "unique_successful_generation_keys": len(unique_successful_keys),
        "failed_generation_records": len(failed),
        "failed_generation_attempt_records": len(failed),
        "unique_attempted_keys": len(unique_attempted_keys),
        "retry_attempt_records": len(records) - len(unique_attempted_keys),
        "expected_generation_records": expected_successful_records,
        "missing_generation_records": missing_successful,
        "missing_successful_generation_records": missing_successful,
    }


def expected_candidate_keys_for_source(manifest: dict, run_config: dict) -> set[str]:
    """Reconstruct a generation shard's exact prediction-independent key set."""

    protocol = run_config["protocol"]
    if "max_objects" not in protocol:
        raise KeyError("candidate run_config protocol lacks max_objects")
    objects = select_pilot_objects(
        manifest["objects"],
        protocol["max_objects"],
        shard_index=int(protocol["shard_index"]),
        num_shards=int(protocol["num_shards"]),
    )
    prompt_methods = list(protocol["prompt_methods"])
    if not prompt_methods or len(prompt_methods) != len(set(prompt_methods)):
        raise RuntimeError("candidate run_config has empty or duplicate prompt methods")
    return {
        candidate_key(item, expression, prompt_method)
        for item in objects
        for expression in item["expressions"]
        for prompt_method in prompt_methods
    }


def run_evaluation(args) -> int:
    source_run_dirs = [Path(args.run_dir).resolve()] + [
        Path(path).resolve() for path in args.additional_run_dir
    ]
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else source_run_dirs[0]
    )
    manifest = json.loads(Path(args.manifest).read_text())
    object_lookup = {
        (item["dataset"], item["video_id"], item["object_id"]): item
        for item in manifest["objects"]
    }
    all_records = []
    source_status = []
    expected_keys_by_source = []
    for source_run_dir in source_run_dirs:
        status_path = source_run_dir / "STATUS.json"
        run_config_path = source_run_dir / "run_config.json"
        run_config = (
            json.loads(run_config_path.read_text())
            if run_config_path.is_file()
            else None
        )
        expected_keys = (
            expected_candidate_keys_for_source(manifest, run_config)
            if run_config is not None
            and "max_objects" in run_config.get("protocol", {})
            else None
        )
        expected_keys_by_source.append(expected_keys)
        source_status.append(
            {
                "run_dir": str(source_run_dir),
                "run_config": str(run_config_path) if run_config_path.is_file() else None,
                "exact_expected_key_count": (
                    len(expected_keys) if expected_keys is not None else None
                ),
                "status": json.loads(status_path.read_text())
                if status_path.is_file()
                else None,
            }
        )
        source_records = []
        for record in load_records(source_run_dir / "candidate_records.jsonl"):
            record = dict(record)
            record["_source_run_dir"] = str(source_run_dir)
            source_records.append(record)
            all_records.append(record)
        if expected_keys is not None:
            extra_attempted = sorted(
                {record["key"] for record in source_records} - expected_keys
            )
            if extra_attempted:
                raise RuntimeError(
                    f"candidate shard contains unexpected keys: {source_run_dir}: "
                    + ", ".join(extra_attempted[:10])
                )
    records = [record for record in all_records if record.get("status") == "success"]
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate successful candidate keys across input shards")
    progress_path = output_dir / "evaluation_progress.json"
    evaluation_started = time.monotonic()
    rows_by_index = [None] * len(records)
    atomic_json(
        progress_path,
        {
            "state": "running",
            "pid": os.getpid(),
            "planned": len(records),
            "completed": 0,
            "current_key": None,
            "workers": args.workers,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "updated_at": utc_now(),
        },
    )
    tasks = [
        (
            record,
            object_lookup[(record["dataset"], record["video_id"], record["object_id"])],
        )
        for record in records
    ]
    executor = None
    completed_count = 0
    current_record = None

    def update_running_progress(record: dict) -> None:
        elapsed = time.monotonic() - evaluation_started
        atomic_json(
            progress_path,
            {
                "state": "running",
                "pid": os.getpid(),
                "planned": len(records),
                "completed": completed_count,
                "current_key": record["key"],
                "workers": args.workers,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed / completed_count * (len(records) - completed_count)
                ),
                "updated_at": utc_now(),
            },
        )

    try:
        if args.workers == 1:
            for row_index, (record, task) in enumerate(zip(records, tasks)):
                current_record = record
                rows_by_index[row_index] = evaluate_record_task(task)
                completed_count += 1
                update_running_progress(record)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            future_lookup = {
                executor.submit(evaluate_record_task, task): (row_index, record)
                for row_index, (record, task) in enumerate(zip(records, tasks))
            }
            for future in as_completed(future_lookup):
                row_index, record = future_lookup[future]
                current_record = record
                rows_by_index[row_index] = future.result()
                completed_count += 1
                update_running_progress(record)
    except Exception as error:
        atomic_json(
            progress_path,
            {
                "state": "failed",
                "pid": os.getpid(),
                "planned": len(records),
                "completed": completed_count,
                "current_key": current_record["key"] if current_record else None,
                "elapsed_seconds": time.monotonic() - evaluation_started,
                "estimated_remaining_seconds": None,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            },
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if completed_count != len(records) or any(row is None for row in rows_by_index):
        raise RuntimeError("candidate evaluation completed with missing result rows")
    rows = list(rows_by_index)
    summary = aggregate_candidate_rows(rows)
    write_csv(output_dir / "sam31_candidate_metrics.csv", rows)
    write_csv(output_dir / "sam31_candidate_summary.csv", summary)
    source_states = [
        source["status"].get("state")
        for source in source_status
        if source["status"] is not None
    ]
    planned_values = [
        int(source["status"]["planned"])
        for source in source_status
        if source["status"] is not None
        and source["status"].get("planned") is not None
    ]
    expected_generation_records = (
        sum(planned_values) if len(planned_values) == len(source_run_dirs) else None
    )
    exact_expected_keys = None
    if all(value is not None for value in expected_keys_by_source):
        expected_key_total = sum(len(value) for value in expected_keys_by_source)
        exact_expected_keys = set().union(*expected_keys_by_source)
        if len(exact_expected_keys) != expected_key_total:
            raise RuntimeError("candidate source shards have overlapping expected key sets")
        if (
            expected_generation_records is not None
            and expected_generation_records != len(exact_expected_keys)
        ):
            raise RuntimeError(
                "candidate STATUS planned totals differ from exact expected keys: "
                f"{expected_generation_records} != {len(exact_expected_keys)}"
            )
        expected_generation_records = len(exact_expected_keys)
    attempt_summary = summarize_generation_attempts(
        all_records, expected_generation_records
    )
    attempted_keys = {record["key"] for record in all_records}
    successful_keys = {
        record["key"]
        for record in all_records
        if record.get("status") == "success"
    }
    failed_attempt_keys = {
        record["key"]
        for record in all_records
        if record.get("status") != "success"
    }
    exact_key_audit = None
    completeness = None
    if exact_expected_keys is not None:
        exact_key_audit = {
            "status": "pass",
            "expected_keys": len(exact_expected_keys),
            "attempted_keys": len(attempted_keys),
            "successful_keys": len(successful_keys),
            "never_attempted_keys": sorted(exact_expected_keys - attempted_keys),
            "missing_successful_keys": sorted(exact_expected_keys - successful_keys),
            "failed_without_successful_retry_keys": sorted(
                failed_attempt_keys - successful_keys
            ),
            "failed_then_successful_retry_keys": sorted(
                failed_attempt_keys & successful_keys
            ),
            "unexpected_attempted_keys": sorted(attempted_keys - exact_expected_keys),
            "source_expected_key_counts": [
                len(value) for value in expected_keys_by_source
            ],
        }
        completeness = build_candidate_completeness(
            manifest, exact_expected_keys, successful_keys
        )
        write_csv(output_dir / "sam31_candidate_completeness.csv", completeness)
        write_csv(
            output_dir / "sam31_candidate_summary_complete_objects.csv",
            aggregate_complete_object_candidate_rows(rows, completeness),
        )
    atomic_json(
        output_dir / "evaluation_status.json",
        {
            "created_at": utc_now(),
            "source_runs": source_status,
            **attempt_summary,
            "exact_expected_key_audit": exact_key_audit,
            "all_source_runs_complete": len(source_states) == len(source_run_dirs)
            and all(
                state in {"complete", "complete_with_failures"}
                for state in source_states
            ),
            "unique_successful_keys": len(set(keys)),
            "evaluated_records": len(rows),
            "evaluation_workers": args.workers,
            "output_dir": str(output_dir),
            "ground_truth_used_only_in_evaluation": True,
        },
    )
    atomic_json(
        progress_path,
        {
            "state": "complete",
            "pid": os.getpid(),
            "planned": len(records),
            "completed": len(rows),
            "current_key": None,
            "workers": args.workers,
            "elapsed_seconds": time.monotonic() - evaluation_started,
            "estimated_remaining_seconds": 0.0,
            "updated_at": utc_now(),
        },
    )
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
    evaluation.add_argument("--additional-run-dir", action="append", default=[])
    evaluation.add_argument("--output-dir")
    evaluation.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "generate":
        raise SystemExit(run_generation(arguments))
    if arguments.workers < 1:
        raise SystemExit("--workers must be positive")
    raise SystemExit(run_evaluation(arguments))

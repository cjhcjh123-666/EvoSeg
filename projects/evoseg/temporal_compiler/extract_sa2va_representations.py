"""Extract Sa2VA [SEG] representations without running the pixel decoder.

The inputs are joined against the completed temporal-segmentation run.  This is
deliberately a representation-only runner: no segmentation frames or ground
truth masks are loaded, and the SAM2 grounding encoder is never called.
"""

from __future__ import annotations

import argparse
import hashlib
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
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


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


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def stable_key(item: dict, expression: dict, budget: int) -> str:
    return "/".join(
        [
            item["dataset"],
            item["video_id"],
            item["object_id"],
            expression["expression_id"],
            str(budget),
        ]
    )


def load_jsonl_by_key(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                records[value["key"]] = value
    return records


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        record["key"]
        for record in load_jsonl_by_key(path).values()
        if record.get("status") == "success"
    }


def load_rgb(paths: list[Path]) -> list[Image.Image]:
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def audit_loaded_model(model, model_path: Path) -> dict:
    runtime_source = Path(inspect.getfile(type(model))).resolve()
    forbidden_fragments = (
        "temporal_existence",
        "existence_head",
        "temporal_gate",
        "frame_gate",
    )
    forbidden_parameters = [
        name
        for name, _ in model.named_parameters()
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    ]
    forbidden_attributes = [
        name
        for name in (
            "temporal_existence_head",
            "existence_head",
            "temporal_gate_enabled",
        )
        if hasattr(model, name)
    ]
    temporal_files = sorted(str(path) for path in model_path.glob("*temporal*head*"))
    source_text = runtime_source.read_text()
    if (
        forbidden_parameters
        or forbidden_attributes
        or temporal_files
        or "load_temporal_head" in source_text
        or "temporal_gate" in source_text
    ):
        raise RuntimeError(
            "temporal head/gate detected; refusing extraction: "
            f"params={forbidden_parameters}, attrs={forbidden_attributes}, "
            f"files={temporal_files}, source={runtime_source}"
        )
    return {
        "runtime_model_source": str(runtime_source),
        "runtime_model_source_sha256": sha256(runtime_source),
        "runtime_model_class": f"{type(model).__module__}.{type(model).__name__}",
        "forbidden_parameters": forbidden_parameters,
        "forbidden_attributes": forbidden_attributes,
        "temporal_head_files": temporal_files,
        "gate_audit_passed": True,
    }


class RepresentationExtractor:
    def __init__(self, model, processor, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.seg_token_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")

    def prepare(self, frames: list[Image.Image], query: str):
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        processed_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[processed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.model.min_pixels,
            max_pixels=self.model.max_pixels,
        ).to(self.model.device)
        input_ids = inputs.input_ids[0]
        image_token_id = int(self.model.config.image_token_id)
        video_token_id = int(self.model.config.video_token_id)
        token_info = {
            "input_tokens": int(input_ids.numel()),
            "visual_tokens": int(
                ((input_ids == image_token_id) | (input_ids == video_token_id))
                .sum()
                .item()
            ),
            "image_grid_thw": (
                inputs.image_grid_thw.detach().cpu().tolist()
                if "image_grid_thw" in inputs
                else None
            ),
            "pixel_values_shape": (
                list(inputs.pixel_values.shape) if "pixel_values" in inputs else None
            ),
        }
        if token_info["image_grid_thw"] is not None:
            assert len(token_info["image_grid_thw"]) == len(frames)
        return inputs, token_info

    @torch.inference_mode()
    def extract(self, frames: list[Image.Image], query: str) -> dict:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inputs, token_info = self.prepare(frames, query)
        generated = self.model.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated.sequences)
        ]
        text_output = self.processor.batch_decode(
            trimmed, skip_special_tokens=False
        )[0].strip()
        # This is the exact alignment used by the completed segmentation run:
        # concatenate the last layer from each generation step, drop the final
        # token id, then select the generated [SEG] position.
        last_layer = torch.cat(
            [step_hidden_states[-1][0] for step_hidden_states in generated.hidden_states],
            dim=0,
        )
        output_ids = generated.sequences[0][:-1]
        output_length = len(output_ids)
        seg_mask = output_ids == self.seg_token_id
        h_seg = last_layer[-output_length:][seg_mask]
        z_seg = self.model.text_hidden_fcs(h_seg)
        torch.cuda.synchronize()
        token_info.update(
            {
                "generated_tokens": int(trimmed[0].numel()),
                "seg_token_count": int(h_seg.shape[0]),
            }
        )
        return {
            "h_seg": h_seg.detach().float().cpu().numpy(),
            "z_seg": z_seg.detach().float().cpu().numpy(),
            "text_output": text_output,
            "token_info": token_info,
            "latency_seconds_synchronized": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }


def select_objects(objects: list[dict], max_videos: int | None, max_objects: int | None):
    if max_videos is not None:
        video_ids: list[str] = []
        selected = []
        for item in objects:
            if item["video_id"] not in video_ids:
                if len(video_ids) >= max_videos:
                    continue
                video_ids.append(item["video_id"])
            selected.append(item)
        objects = selected
    if max_objects is not None:
        objects = objects[:max_objects]
    return objects


def merge_status(path: Path, **updates) -> dict:
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    atomic_json(path, current)
    return current


def run(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    vector_dir = run_dir / "representations"
    log_dir = run_dir / "logs"
    vector_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    previous_path = Path(args.previous_predictions).resolve()
    model_path = Path(args.model).resolve()
    manifest = json.loads(manifest_path.read_text())
    previous = load_jsonl_by_key(previous_path)
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, {args.num_shards}), got {args.shard_rank}"
        )
    base_records_path = run_dir / "representation_records.jsonl"
    records_path = (
        base_records_path
        if args.num_shards == 1
        else run_dir
        / f"representation_records.worker-{args.shard_rank:02d}-of-{args.num_shards:02d}.jsonl"
    )
    # A sharded continuation inherits the successes produced by the preceding
    # single-worker run, then records only its own deterministic partition.
    done = completed_keys(base_records_path)
    if records_path != base_records_path:
        done.update(completed_keys(records_path))
    status_path = (
        run_dir / "STATUS.json"
        if args.num_shards == 1
        else run_dir
        / f"STATUS.worker-{args.shard_rank:02d}-of-{args.num_shards:02d}.json"
    )
    merge_status(
        status_path,
        state="loading_model",
        phase=args.phase,
        pid=os.getpid(),
        updated_at=utc_now(),
        completed=len(done),
    )

    torch.cuda.set_device(args.device)
    load_started = time.perf_counter()
    model, loading_info = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        output_loading_info=True,
    )
    model = model.eval().to(f"cuda:{args.device}")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    torch.cuda.synchronize()
    audit = audit_loaded_model(model, model_path)
    extractor = RepresentationExtractor(model, processor, args.max_new_tokens)
    loading_summary = {
        key: value
        for key, value in loading_info.items()
        if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    }
    config = {
        "schema_version": 1,
        "created_at": utc_now(),
        "code_commit": git_output("rev-parse", "HEAD"),
        "code_dirty": bool(git_output("status", "--porcelain")),
        "branch": git_output("branch", "--show-current"),
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "model_index_sha256": sha256(model_path / "model.safetensors.index.json"),
        "model_load_seconds_synchronized": time.perf_counter() - load_started,
        "loading_info": loading_summary,
        "runtime_audit": audit,
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu_index": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
        },
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "previous_predictions": str(previous_path),
            "previous_predictions_sha256": sha256(previous_path),
            "frame_budgets": args.frame_budgets,
            "query_template": "Please segment {official_expression} in this video.",
            "segmentation_invoked": False,
            "ground_truth_loaded": False,
            "max_new_tokens": args.max_new_tokens,
            "vlm_min_pixels": model.min_pixels,
            "vlm_max_pixels": model.max_pixels,
            "num_shards": args.num_shards,
            "shard_rank": args.shard_rank,
        },
    }
    config_path = (
        run_dir / "run_config.json"
        if args.num_shards == 1
        else run_dir
        / f"run_config.worker-{args.shard_rank:02d}-of-{args.num_shards:02d}.json"
    )
    atomic_json(config_path, config)

    objects = select_objects(manifest["objects"], args.max_videos, args.max_objects)
    all_jobs = [
        (item, expression, budget)
        for item in objects
        for expression in item["expressions"]
        for budget in args.frame_budgets
    ]
    jobs = [
        job
        for job_index, job in enumerate(all_jobs)
        if job_index % args.num_shards == args.shard_rank
    ]
    planned = len(jobs)
    pending = sum(
        stable_key(item, expression, budget) not in done
        for item, expression, budget in jobs
    )
    started = time.monotonic()
    attempted = 0
    failed = 0
    merge_status(
        status_path,
        state="running",
        planned=planned,
        pending_at_start=pending,
        attempted_this_process=0,
        failed_this_process=0,
        updated_at=utc_now(),
    )
    for item, expression, budget in jobs:
        key = stable_key(item, expression, budget)
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
            "frame_budget": budget,
            "vlm_frame_indices": item["vlm_frame_indices"][str(budget)],
            "vlm_frame_names": item["vlm_frame_names"][str(budget)],
            "segmentation_invoked": False,
            "ground_truth_loaded": False,
            "started_at": utc_now(),
        }
        try:
            if key not in previous:
                raise KeyError(f"key absent from completed prior run: {key}")
            prior = previous[key]
            expected = {
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "description_type": expression["type"],
                "expression": expression["text"],
                "frame_budget": budget,
                "vlm_frame_indices": item["vlm_frame_indices"][str(budget)],
            }
            mismatches = {
                field: (prior.get(field), value)
                for field, value in expected.items()
                if prior.get(field) != value
            }
            if mismatches:
                raise AssertionError(f"prior-run input mismatch: {mismatches}")
            frame_paths = [
                Path(manifest["dataset"]["image_root"])
                / item["video_id"]
                / f"{name}.jpg"
                for name in item["vlm_frame_names"][str(budget)]
            ]
            frames = load_rgb(frame_paths)
            if len(frames) != min(budget, item["frame_count"]):
                raise AssertionError((len(frames), budget, item["frame_count"]))
            output = extractor.extract(
                frames,
                f"Please segment {expression['text']} in this video.",
            )
            if output["token_info"]["seg_token_count"] != 1:
                raise AssertionError(
                    "main analysis requires exactly one [SEG] token, got "
                    f"{output['token_info']['seg_token_count']}"
                )
            if output["text_output"] != prior["prediction"]:
                raise AssertionError(
                    f"generated text changed: {output['text_output']!r} != "
                    f"{prior['prediction']!r}"
                )
            if output["token_info"]["visual_tokens"] != prior["token_info"][
                "visual_tokens"
            ]:
                raise AssertionError(
                    "visual token count changed: "
                    f"{output['token_info']['visual_tokens']} != "
                    f"{prior['token_info']['visual_tokens']}"
                )
            vector_name = key.replace("/", "__") + ".npz"
            vector_path = vector_dir / vector_name
            np.savez_compressed(
                vector_path,
                h_seg=output.pop("h_seg"),
                z_seg=output.pop("z_seg"),
            )
            base.update(
                {
                    "status": "success",
                    "completed_at": utc_now(),
                    "vector_path": str(vector_path.relative_to(run_dir)),
                    "vector_sha256": sha256(vector_path),
                    **output,
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
        append_jsonl(records_path, base)
        attempted += 1
        elapsed = time.monotonic() - started
        remaining = max(pending - attempted, 0)
        merge_status(
            status_path,
            state="running",
            updated_at=utc_now(),
            completed=len(done),
            attempted_this_process=attempted,
            failed_this_process=failed,
            elapsed_seconds_this_process=elapsed,
            estimated_remaining_seconds=(
                elapsed / attempted * remaining if attempted else None
            ),
            current_key=key,
        )
    merge_status(
        status_path,
        state="phase_complete" if failed == 0 else "phase_complete_with_failures",
        updated_at=utc_now(),
        completed=len(done),
        attempted_this_process=attempted,
        failed_this_process=failed,
        elapsed_seconds_this_process=time.monotonic() - started,
        estimated_remaining_seconds=0.0,
        current_key=None,
    )
    return 0 if failed == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--previous-predictions", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model",
        default="/9950backfile/chenjiahui/evo_artifacts/models/Sa2VA-Qwen3-VL-4B",
    )
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--frame-budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--phase", choices=["smoke", "protocol64", "full"], default="full")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

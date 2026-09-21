"""Resumable Sa2VA runner with separated VLM and segmentation frame streams.

This runner intentionally does not call the legacy ``infer_ryvos.py`` or the
checkpoint's ``predict_forward``.  It uses the original Sa2VA modules and
weights, but makes the two frame streams explicit and fails loudly if a
temporal/existence gate is present.
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
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .official_metrics import db_eval_boundary, db_eval_iou


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_sha256(path: str | Path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def stable_key(item, expression, frame_budget):
    return "/".join(
        [
            item["dataset"],
            item["video_id"],
            item["object_id"],
            expression["expression_id"],
            str(frame_budget),
        ]
    )


def load_rgb(paths):
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def encode_masks(masks, path: Path, frame_names):
    encoded = []
    for frame_name, mask in zip(frame_names, masks):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        encoded.append({"frame_name": frame_name, "rle": rle})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encoded, separators=(",", ":")) + "\n")


def load_ground_truth(item, shape):
    gt = []
    for path in item["evaluation_mask_paths"]:
        if Path(path).is_file():
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.shape != shape:
                raise ValueError(f"GT shape {mask.shape} differs from prediction {shape}: {path}")
        else:
            # This matches the Long-RVOS official evaluator: an absent PNG is
            # an empty target frame, not a missing sample.
            mask = np.zeros(shape, dtype=bool)
        gt.append(mask)
    return np.stack(gt)


class SeparatedSa2VA:
    def __init__(self, model, tokenizer, processor, max_new_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.seg_token_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")

    def _prepare_vlm(self, frames, query):
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
                ((input_ids == image_token_id) | (input_ids == video_token_id)).sum().item()
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
            assert len(token_info["image_grid_thw"]) == len(frames), (
                len(token_info["image_grid_thw"]),
                len(frames),
            )
        return inputs, token_info

    def _prepare_segmentation(self, frames):
        sizes = {frame.size for frame in frames}
        if len(sizes) != 1:
            raise ValueError(f"segmentation frames have inconsistent sizes: {sizes}")
        values = []
        for frame in frames:
            image = self.model.extra_image_processor.apply_image(np.asarray(frame))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            values.append(self.model.grounding_encoder.preprocess_image(image))
        return torch.stack(values).to(self.model.device, dtype=self.model.torch_dtype)

    def _sam2_masks(self, segmentation_values, embedding, prompt_positions):
        state = self.model.grounding_encoder.get_sam2_embeddings(segmentation_values)
        sam2_model = self.model.grounding_encoder.sam2_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for position in prompt_positions:
                language = embedding.unsqueeze(0).unsqueeze(0)
                sam2_model.add_language_embd(
                    state, int(position), 100, language, inference=True
                )
            by_index = {}
            for index, _object_ids, logits in sam2_model.propagate_in_video(state):
                by_index[int(index)] = logits
        expected = set(range(segmentation_values.shape[0]))
        if set(by_index) != expected:
            raise RuntimeError(
                f"SAM2 returned frame indices {sorted(by_index)}; expected {sorted(expected)}"
            )
        return torch.cat([by_index[i] for i in range(len(by_index))], dim=0)

    @torch.inference_mode()
    def predict(self, vlm_frames, segmentation_frames, text, prompt_positions):
        """No ground truth argument exists: GT cannot enter model inference."""
        assert vlm_frames and segmentation_frames
        assert prompt_positions == list(range(len(prompt_positions)))
        assert max(prompt_positions, default=-1) < len(segmentation_frames)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        mm_inputs, token_info = self._prepare_vlm(vlm_frames, text)
        segmentation_values = self._prepare_segmentation(segmentation_frames)
        generated = self.model.model.generate(
            **mm_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(mm_inputs.input_ids, generated.sequences)
        ]
        prediction = self.processor.batch_decode(
            trimmed, skip_special_tokens=False
        )[0].strip()
        hidden = torch.cat([step[-1][0] for step in generated.hidden_states], dim=0)
        output_ids = generated.sequences[0][:-1]
        output_length = len(output_ids)
        seg_mask = output_ids == self.seg_token_id
        seg_hidden = hidden[-output_length:][seg_mask] if output_length else hidden[0:0]
        embeddings = self.model.text_hidden_fcs(seg_hidden)
        width, height = segmentation_frames[0].size
        mask_sets = []
        for embedding in embeddings:
            logits = self._sam2_masks(
                segmentation_values, embedding, prompt_positions
            )
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            mask_sets.append((logits[:, 0].sigmoid() > 0.5).cpu().numpy())
        if mask_sets:
            masks = mask_sets[0]
        else:
            masks = np.zeros(
                (len(segmentation_frames), height, width), dtype=bool
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        token_info.update(
            {
                "generated_tokens": int(trimmed[0].numel()),
                "seg_token_count": int(embeddings.shape[0]),
            }
        )
        return {
            "prediction": prediction,
            "masks": masks,
            "token_info": token_info,
            "latency_seconds_synchronized": latency,
            "peak_memory_bytes": int(peak),
        }


def audit_loaded_model(model, model_path):
    runtime_source = Path(inspect.getfile(type(model))).resolve()
    forbidden_parameters = [
        name
        for name, _ in model.named_parameters()
        if any(word in name.lower() for word in ("temporal_existence", "existence_head", "gate"))
    ]
    forbidden_attributes = [
        name
        for name in ("temporal_existence_head", "existence_head", "temporal_gate_enabled")
        if hasattr(model, name)
    ]
    temporal_files = sorted(str(p) for p in Path(model_path).glob("*temporal*head*"))
    if forbidden_parameters or forbidden_attributes or temporal_files:
        raise RuntimeError(
            "temporal head/gate detected; refusing inference: "
            f"params={forbidden_parameters}, attrs={forbidden_attributes}, files={temporal_files}"
        )
    source_text = runtime_source.read_text()
    if "temporal_gate" in source_text or "load_temporal_head" in source_text:
        raise RuntimeError(f"runtime source contains temporal gate code: {runtime_source}")
    return {
        "runtime_model_source": str(runtime_source),
        "runtime_model_source_sha256": file_sha256(runtime_source),
        "runtime_model_class": f"{type(model).__module__}.{type(model).__name__}",
        "forbidden_parameters": forbidden_parameters,
        "forbidden_attributes": forbidden_attributes,
        "temporal_head_files": temporal_files,
        "gate_audit_passed": True,
    }


def completed_keys(path: Path):
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") in {"success", "success_no_seg"}:
            keys.add(record["key"])
    return keys


def append_record(path: Path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args):
    run_dir = Path(args.run_dir).resolve()
    for name in ("logs", "masks", "figures"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    predictions_path = run_dir / "predictions.jsonl"
    done = completed_keys(predictions_path)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({
        "updated_at": utc_now(),
        "state": "loading_model",
        "pid": os.getpid(),
        "phase": args.phase,
        "completed_unique_results": len(done),
        "failed_results": 0,
        "blockers": [],
    })
    atomic_json(status_path, status)

    torch.cuda.set_device(args.device)
    load_started = time.perf_counter()
    loaded = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
    model = model.eval().to(f"cuda:{args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    audit = audit_loaded_model(model, args.model)
    runtime = SeparatedSa2VA(model, tokenizer, processor, args.max_new_tokens)

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
        "model_path": str(Path(args.model).resolve()),
        "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
        "model_index_sha256": file_sha256(Path(args.model) / "model.safetensors.index.json"),
        "checkpoint_choice": "original Sa2VA-Qwen3-VL-4B",
        "checkpoint_fallback_used": False,
        "model_load_seconds_synchronized": load_seconds,
        "loading_info": loading_summary,
        "runtime_audit": audit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu_index": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
        },
        "protocol": {
            "manifest": str(Path(args.manifest).resolve()),
            "frame_budgets": args.frame_budgets,
            "only_vlm_frames_change_across_budgets": True,
            "segmentation_frames_from_manifest": True,
            "sam2_prompt_positions_from_manifest": True,
            "ground_truth_passed_to_model": False,
            "resolution_rule": {
                "vlm_min_pixels": model.min_pixels,
                "vlm_max_pixels": model.max_pixels,
                "segmentation_square_pixels": 1024,
            },
            "max_new_tokens": args.max_new_tokens,
            "metric_source": (
                "iSEE-Laboratory/Long_RVOS@"
                "447682a4fba314e81897645a91ff1a178da493d3"
            ),
        },
    }
    atomic_json(run_dir / "run_config.json", config)

    objects = manifest["objects"]
    if args.max_videos:
        allowed = []
        video_ids = []
        for item in objects:
            if item["video_id"] not in video_ids:
                if len(video_ids) >= args.max_videos:
                    continue
                video_ids.append(item["video_id"])
            allowed.append(item)
        objects = allowed
    total = sum(len(item["expressions"]) * len(args.frame_budgets) for item in objects)
    status.update({"state": "running", "planned_results_this_phase": total})
    atomic_json(status_path, status)

    first_inference = not done
    failed = 0
    for item in objects:
        segmentation_names = item["segmentation_frame_names"]
        segmentation_paths = [
            str(
                Path(manifest["dataset"]["image_root"])
                / item["video_id"]
                / f"{name}.jpg"
            )
            for name in segmentation_names
        ]
        segmentation_frames = load_rgb(segmentation_paths)
        segmentation_signature = hashlib.sha256(
            json.dumps(segmentation_names).encode()
        ).hexdigest()
        for expression in item["expressions"]:
            for budget in args.frame_budgets:
                key = stable_key(item, expression, budget)
                if key in done:
                    continue
                base_record = {
                    "key": key,
                    "dataset": item["dataset"],
                    "split": item["split"],
                    "video_id": item["video_id"],
                    "object_id": item["object_id"],
                    "expression_id": expression["expression_id"],
                    "description_type": expression["type"],
                    "expression": expression["text"],
                    "description_length_chars": expression["length_chars"],
                    "description_length_words": expression["length_words"],
                    "frame_budget": budget,
                    "vlm_frame_indices": item["vlm_frame_indices"][str(budget)],
                    "vlm_frame_names": item["vlm_frame_names"][str(budget)],
                    "segmentation_frame_indices": item["segmentation_frame_indices"],
                    "segmentation_frame_names": segmentation_names,
                    "evaluation_frame_indices": item["evaluation_frame_indices"],
                    "evaluation_frame_names": item["evaluation_frame_names"],
                    "sam2_prompt_positions": item["sam2_prompt_positions"],
                    "segmentation_frame_signature": segmentation_signature,
                    "gt_entered_model": False,
                    "phase_first_seen": args.phase,
                    "started_at": utc_now(),
                    "thermal_state": "cold_first_inference" if first_inference else "warm",
                }
                try:
                    vlm_paths = [
                        str(
                            Path(manifest["dataset"]["image_root"])
                            / item["video_id"]
                            / f"{name}.jpg"
                        )
                        for name in item["vlm_frame_names"][str(budget)]
                    ]
                    vlm_frames = load_rgb(vlm_paths)
                    assert len(vlm_frames) == min(budget, item["frame_count"])
                    output = runtime.predict(
                        vlm_frames,
                        segmentation_frames,
                        f"Please segment {expression['text']} in this video.",
                        item["sam2_prompt_positions"],
                    )
                    masks = output.pop("masks")
                    assert len(masks) == len(segmentation_frames)
                    gt = load_ground_truth(item, masks.shape[-2:])
                    j_values = db_eval_iou(gt, masks)
                    f_values = db_eval_boundary(gt, masks)
                    safe_key = key.replace("/", "__")
                    relative_masks = Path("masks") / f"{safe_key}.json"
                    encode_masks(masks, run_dir / relative_masks, segmentation_names)
                    base_record.update(
                        {
                            "status": (
                                "success"
                                if output["token_info"]["seg_token_count"]
                                else "success_no_seg"
                            ),
                            "completed_at": utc_now(),
                            "J": float(np.mean(j_values)),
                            "F": float(np.mean(f_values)),
                            "J_and_F": float((np.mean(j_values) + np.mean(f_values)) / 2),
                            "per_frame_J": j_values.tolist(),
                            "per_frame_F": f_values.tolist(),
                            "rle_masks_path": str(relative_masks),
                            **output,
                        }
                    )
                    done.add(key)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    failed += 1
                    base_record.update(
                        {
                            "status": "failed_oom",
                            "completed_at": utc_now(),
                            "error": str(exc),
                        }
                    )
                except Exception as exc:
                    failed += 1
                    base_record.update(
                        {
                            "status": "failed",
                            "completed_at": utc_now(),
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                append_record(predictions_path, base_record)
                first_inference = False
                status.update(
                    {
                        "updated_at": utc_now(),
                        "state": "running",
                        "completed_unique_results": len(done),
                        "failed_results_this_process": failed,
                        "current_key": key,
                    }
                )
                atomic_json(status_path, status)
    status.update(
        {
            "updated_at": utc_now(),
            "state": "phase_complete",
            "completed_unique_results": len(done),
            "failed_results_this_process": failed,
            "current_key": None,
        }
    )
    atomic_json(status_path, status)
    return 0 if failed == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model",
        default="/9950backfile/chenjiahui/evo_artifacts/models/Sa2VA-Qwen3-VL-4B",
    )
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--frame-budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

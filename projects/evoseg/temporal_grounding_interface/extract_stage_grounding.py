"""Extract stage-specific Sa2VA states and anchor grounding masks.

This runner never opens Long-RVOS annotations.  For every expression it makes
two parameter-identical frozen Sa2VA passes at each canonical stage: a temporal
pass over the uniformly sampled observed prefix, and a static control with the
same number of image slots all containing the current anchor frame.  The
resulting z state is decoded only on the anchor image and cached for later
candidate scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from projects.evoseg.temporal_grounding_interface.protocol import (
    anchor_only_positions,
    cumulative_visible_positions,
    stable_identity,
    stage_end_positions,
)
from projects.evoseg.temporal_seg.runner import audit_loaded_model, file_sha256


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


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


class StageGroundingRuntime:
    def __init__(self, model, processor, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.seg_token_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")

    def _prepare_vlm(self, frames: list[Image.Image], query: str):
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.model.min_pixels,
            max_pixels=self.model.max_pixels,
        ).to(self.model.device)
        ids = inputs.input_ids[0]
        image_token = int(self.model.config.image_token_id)
        video_token = int(self.model.config.video_token_id)
        return inputs, {
            "input_tokens": int(ids.numel()),
            "visual_tokens": int(((ids == image_token) | (ids == video_token)).sum()),
            "image_grid_thw": inputs.image_grid_thw.detach().cpu().tolist()
            if "image_grid_thw" in inputs
            else None,
        }

    def _anchor_tensor(self, frame: Image.Image):
        value = self.model.extra_image_processor.apply_image(np.asarray(frame))
        value = torch.from_numpy(value).permute(2, 0, 1).contiguous()
        value = self.model.grounding_encoder.preprocess_image(value)
        return value[None].to(self.model.device, dtype=self.model.torch_dtype)

    def _decode_anchor(self, frame: Image.Image, embedding: torch.Tensor) -> np.ndarray:
        values = self._anchor_tensor(frame)
        state = self.model.grounding_encoder.get_sam2_embeddings(values)
        sam = self.model.grounding_encoder.sam2_model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            language = embedding[None, None]
            _frame, _ids, logits = sam.add_language_embd(
                state, 0, 100, language, inference=True
            )
            # The direct correction output is the current-frame grounding mask;
            # propagation over a one-frame state is unnecessary and less explicit.
            logits = F.interpolate(
                logits, size=(frame.height, frame.width), mode="bilinear", align_corners=False
            )
        return (logits[0, 0].sigmoid() > 0.5).cpu().numpy()

    @torch.inference_mode()
    def run(self, frames: list[Image.Image], anchor: Image.Image, query: str) -> dict:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inputs, token_info = self._prepare_vlm(frames, query)
        generated = self.model.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated.sequences)
        ]
        output_ids = generated.sequences[0][:-1]
        hidden = torch.cat(
            [step_hidden[-1][0] for step_hidden in generated.hidden_states], dim=0
        )
        output_length = len(output_ids)
        seg_mask = output_ids == self.seg_token_id
        h_seg = hidden[-output_length:][seg_mask]
        z_seg = self.model.text_hidden_fcs(h_seg)
        if z_seg.shape[0] != 1:
            raise RuntimeError(f"expected one [SEG] state, received {z_seg.shape[0]}")
        mask = self._decode_anchor(anchor, z_seg[0])
        torch.cuda.synchronize()
        token_info.update(
            {
                "generated_tokens": int(trimmed[0].numel()),
                "seg_token_count": int(z_seg.shape[0]),
            }
        )
        return {
            "h_seg": h_seg[0].detach().float().cpu().numpy(),
            "z_seg": z_seg[0].detach().float().cpu().numpy(),
            "mask": mask,
            "text_output": self.processor.batch_decode(
                trimmed, skip_special_tokens=False
            )[0].strip(),
            "token_info": token_info,
            "latency_seconds_synchronized": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }


def run(args) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    objects = manifest["objects"][: args.max_objects] if args.max_objects else manifest["objects"]
    objects = [
        item for index, item in enumerate(objects) if index % args.num_shards == args.shard_index
    ]
    run_dir = Path(args.run_dir).resolve()
    records_path = run_dir / "stage_grounding_records.jsonl"
    done = completed(records_path)
    (run_dir / "states").mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
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
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    runtime = StageGroundingRuntime(model, processor, args.max_new_tokens)
    model_audit = audit_loaded_model(model, args.model)
    atomic_json(
        run_dir / "stage_grounding_config.json",
        {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "model": str(Path(args.model).resolve()),
            "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "runtime_audit": model_audit,
            "loading_info": {
                key: value
                for key, value in loading_info.items()
                if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
            },
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "protocol": {
                "canonical_stages": 8,
                "maximum_cumulative_frames": args.maximum_cumulative_frames,
                "static_control": "current anchor repeated to temporal image-slot count",
                "anchor_decoder": "frozen Sa2VA SAM2 current-frame language embedding",
                "ground_truth_read": False,
                "selected_objects": len(objects),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        },
    )
    planned = sum(len(item["expressions"]) for item in objects)
    attempted = failed = 0
    started = time.monotonic()
    image_root = Path(manifest["dataset"]["image_root"])
    for item in objects:
        endpoints = stage_end_positions(item["frame_count"], 8)
        paths = [image_root / item["video_id"] / f"{name}.jpg" for name in item["frame_names"]]
        cache: dict[int, Image.Image] = {}

        def image_at(position: int) -> Image.Image:
            if position not in cache:
                cache[position] = load_rgb(paths[position])
            return cache[position]

        for expression in item["expressions"]:
            identity = stable_identity(
                item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]
            )
            if identity in done:
                continue
            record = {
                "identity": identity,
                "dataset": item["dataset"],
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "description_type": expression["type"],
                "expression": expression["text"],
                "stage_end_indices": endpoints,
                "gt_read_during_inference": False,
                "started_at": utc_now(),
            }
            try:
                arrays = {}
                state_metadata = []
                query = f"Please segment {expression['text']} in this video."
                for stage_index, endpoint in enumerate(endpoints):
                    temporal_positions = cumulative_visible_positions(
                        item["frame_count"], endpoint, args.maximum_cumulative_frames
                    )
                    for state_kind, positions in (
                        ("temporal", temporal_positions),
                        ("static", anchor_only_positions(endpoint, len(temporal_positions))),
                    ):
                        output = runtime.run(
                            [image_at(position) for position in positions], image_at(endpoint), query
                        )
                        prefix = f"{state_kind}_{stage_index}"
                        arrays[f"{prefix}_h"] = output.pop("h_seg")
                        arrays[f"{prefix}_z"] = output.pop("z_seg")
                        arrays[f"{prefix}_mask"] = output.pop("mask").astype(np.uint8)
                        state_metadata.append(
                            {
                                "state_kind": state_kind,
                                "stage_index": stage_index,
                                "stage_end_index": endpoint,
                                "visible_frame_indices": positions,
                                "visible_frame_names": [item["frame_names"][p] for p in positions],
                                **output,
                            }
                        )
                relative = Path("states") / f"{identity.replace('/', '__')}.npz"
                np.savez_compressed(run_dir / relative, **arrays)
                record.update(
                    {
                        "status": "success",
                        "completed_at": utc_now(),
                        "state_path": str(relative),
                        "state_metadata": state_metadata,
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
                run_dir / "STAGE_STATUS.json",
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
        cache.clear()
    atomic_json(
        run_dir / "STAGE_STATUS.json",
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--maximum-cumulative-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))


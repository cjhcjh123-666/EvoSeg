"""Run a real 16-frame Qwen3-VL hidden-state and token-span audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .frame_protocol import gather_frame_tokens, recover_frame_token_spans


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("empty subsequence")
    for start in range(len(sequence) - len(subsequence) + 1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start, start + len(subsequence)
    raise ValueError("query token sequence not found in multimodal input")


def make_audit_frames(count: int = 16, size: int = 336) -> list[Image.Image]:
    """Create deterministic, visually distinct frames without external data."""
    y, x = np.mgrid[:size, :size]
    frames = []
    for index in range(count):
        array = np.stack(
            [
                (x + index * 13) % 256,
                (y * 2 + index * 29) % 256,
                ((x // 2 + y // 3) + index * 47) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        frames.append(Image.fromarray(array, mode="RGB"))
    return frames


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-VL frame audit")
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    model.requires_grad_(False)

    frames = make_audit_frames(args.frames, args.frame_size)
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": args.query})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        min_pixels=args.pixels,
        max_pixels=args.pixels,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1]

    spans = recover_frame_token_spans(
        inputs.input_ids,
        inputs.image_grid_thw,
        list(range(args.frames)),
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        processor.image_processor.merge_size,
    )
    frame_tokens, frame_summaries = gather_frame_tokens(hidden, spans)
    query_ids = processor.tokenizer(args.query, add_special_tokens=False).input_ids
    query_span = find_subsequence(inputs.input_ids[0].tolist(), query_ids)
    query_hidden = hidden[0, query_span[0] : query_span[1]]

    pairwise = torch.nn.functional.cosine_similarity(
        frame_summaries[:-1].float(), frame_summaries[1:].float(), dim=-1
    )
    last_six = range(model.config.text_config.num_hidden_layers - 6, model.config.text_config.num_hidden_layers)
    module_names = []
    for name, module in model.language_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        if any(f"layers.{layer}." in name for layer in last_six):
            module_names.append(name)

    result = {
        "status": "PASS",
        "model": str(args.model),
        "model_type": model.config.model_type,
        "language_layers": model.config.text_config.num_hidden_layers,
        "language_hidden_size": model.config.text_config.hidden_size,
        "attention_implementation": "flash_attention_2",
        "requested_frames": args.frames,
        "actual_frames": len(spans),
        "source_frame_indices": [span.source_frame_index for span in spans],
        "image_grid_thw": inputs.image_grid_thw.cpu().tolist(),
        "frame_token_spans": [
            {
                "frame": span.frame_number,
                "source_frame_index": span.source_frame_index,
                "start": span.token_start,
                "end": span.token_end,
                "tokens": span.token_count,
            }
            for span in spans
        ],
        "visual_token_count": sum(span.token_count for span in spans),
        "tokens_per_frame": [tensor.shape[0] for tensor in frame_tokens],
        "input_sequence_length": inputs.input_ids.shape[-1],
        "query_token_span": list(query_span),
        "query_token_count": len(query_ids),
        "frame_summary_cross_frame_std": frame_summaries.float().std(dim=0).mean().item(),
        "adjacent_frame_cosine_mean": pairwise.mean().item(),
        "adjacent_frame_cosine_min": pairwise.min().item(),
        "query_hidden_norm": query_hidden.float().norm(dim=-1).mean().item(),
        "lora_last_six_target_count": len(module_names),
        "lora_last_six_target_modules": module_names,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if len(spans) != args.frames or len(module_names) != 6 * 4:
        result["status"] = "FAIL"
    if result["frame_summary_cross_frame_std"] <= 0:
        result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-4B-Instruct"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=336)
    parser.add_argument("--pixels", type=int, default=128 * 28 * 28)
    parser.add_argument(
        "--query",
        default="Segment the person who approaches the table and then sits down.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)

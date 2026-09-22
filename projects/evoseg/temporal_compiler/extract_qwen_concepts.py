"""Extract GT-free object concepts from official expressions with Qwen text-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = (
    "Extract only the basic object noun or noun phrase referred to by the expression. "
    "Return only that noun phrase, with no explanation. Do not return a pronoun. "
    "Omit actions, temporal clauses, and spatial relations; retain an attribute only "
    "when it is needed to distinguish the object."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expression_key(item: dict, expression: dict) -> str:
    return "/".join(
        [
            item["dataset"], item["video_id"], str(item["object_id"]),
            str(expression["expression_id"]),
        ]
    )


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def clean_concept(value: str) -> str:
    value = value.strip().splitlines()[0].strip()
    for prefix in ("Concept:", "Object:", "Noun phrase:"):
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix):].strip()
    return value.strip('"\'`').rstrip(".").strip().strip('"\'`')


def run(args) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    config_path = output.with_name("qwen_concept_run_config.json")
    status_path = output.with_name("qwen_concept_STATUS.json")
    existing = {row["key"] for row in load_records(output) if row.get("status") == "success"}

    examples = [
        (item, expression)
        for item in manifest["objects"]
        for expression in item["expressions"]
        if expression_key(item, expression) not in existing
    ]
    torch.cuda.set_device(args.device)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    atomic_json(
        config_path,
        {
            "created_at": utc_now(),
            "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
            "checkpoint_index_sha256": sha256(
                checkpoint / "model.safetensors.index.json"
            ),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "ground_truth_loaded": False,
            "text_only": True,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
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
        },
    )

    failed = 0
    started = time.monotonic()
    for offset in range(0, len(examples), args.batch_size):
        batch = examples[offset:offset + args.batch_size]
        texts = []
        for _, expression in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": expression["text"]},
            ]
            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        try:
            inputs = processor(
                text=texts, padding=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            input_width = inputs.input_ids.shape[1]
            outputs = processor.batch_decode(
                generated[:, input_width:], skip_special_tokens=True
            )
            if len(outputs) != len(batch):
                raise AssertionError("Qwen output batch length mismatch")
            for (item, expression), raw_output in zip(batch, outputs):
                concept = clean_concept(raw_output)
                if not concept:
                    raise ValueError(f"empty Qwen concept for {expression['text']!r}")
                append_jsonl(
                    output,
                    {
                        "key": expression_key(item, expression),
                        "dataset": item["dataset"],
                        "video_id": item["video_id"],
                        "object_id": item["object_id"],
                        "expression_id": expression["expression_id"],
                        "description_type": expression["type"],
                        "expression": expression["text"],
                        "concept": concept,
                        "raw_output": raw_output,
                        "status": "success",
                        "ground_truth_loaded": False,
                        "completed_at": utc_now(),
                    },
                )
                existing.add(expression_key(item, expression))
        except Exception as error:
            failed += len(batch)
            for item, expression in batch:
                append_jsonl(
                    output,
                    {
                        "key": expression_key(item, expression),
                        "dataset": item["dataset"],
                        "video_id": item["video_id"],
                        "object_id": item["object_id"],
                        "expression_id": expression["expression_id"],
                        "description_type": expression["type"],
                        "expression": expression["text"],
                        "status": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "ground_truth_loaded": False,
                        "completed_at": utc_now(),
                    },
                )
        completed = len(existing)
        elapsed = time.monotonic() - started
        atomic_json(
            status_path,
            {
                "state": "running",
                "pid": os.getpid(),
                "planned": sum(len(item["expressions"]) for item in manifest["objects"]),
                "completed": completed,
                "failed_this_process": failed,
                "elapsed_seconds_this_process": elapsed,
                "updated_at": utc_now(),
            },
        )
    planned = sum(len(item["expressions"]) for item in manifest["objects"])
    atomic_json(
        status_path,
        {
            "state": "complete" if len(existing) == planned and failed == 0 else "complete_with_failures",
            "pid": os.getpid(),
            "planned": planned,
            "completed": len(existing),
            "failed_this_process": failed,
            "elapsed_seconds_this_process": time.monotonic() - started,
            "updated_at": utc_now(),
        },
    )
    return 0 if len(existing) == planned and failed == 0 else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

"""Convert the public Qwen3-VL/SAM3 HF export into Sa2VA training weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


def _destination_key(key: str, sam_scope: str) -> str | None:
    if key.startswith("model."):
        return "mllm." + key
    if key.startswith("text_hidden_fcs."):
        return key
    if key.startswith("grounding_encoder."):
        if sam_scope == "all":
            return key
        decoder_prefix = "grounding_encoder.sam2_model.sam_mask_decoder."
        return key if key.startswith(decoder_prefix) else None
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, output: Path, sam_scope: str) -> dict:
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source_key, filename in index["weight_map"].items():
        destination = _destination_key(source_key, sam_scope)
        if destination is not None:
            shards[filename].append((source_key, destination))

    converted = OrderedDict()
    for filename in sorted(shards):
        tensors = load_file(str(source / filename), device="cpu")
        for source_key, destination in sorted(shards[filename]):
            converted[destination] = tensors[source_key]
        del tensors

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(converted, temporary)
    os.replace(temporary, output)
    summary = {
        "source": str(source),
        "source_index_sha256": _sha256(index_path),
        "output": str(output),
        "output_sha256": _sha256(output),
        "sam_scope": sam_scope,
        "tensor_count": len(converted),
        "parameter_count": sum(tensor.numel() for tensor in converted.values()),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam-scope", choices=("all", "decoder"), default="all")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(convert(arguments.source, arguments.output, arguments.sam_scope), indent=2))

"""Minimal frozen-feature temporal track matcher for the SAM3.1 candidate bank.

The extractor never reads ground truth.  It combines the already cached Sa2VA
N=16 query state with frozen InstructSeg SigLIP patch features pooled inside
SAM3.1 candidate masks.  The trainer reads the independently generated oracle
metrics only after feature extraction, uses source-video-disjoint splits, and
compares parameter-identical static and temporally ordered scorers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def expression_identity(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record["dataset"]),
        str(record["video_id"]),
        str(record["object_id"]),
        str(record["expression_id"]),
    )


def representation_key(record: dict, budget: int = 16) -> str:
    return "/".join((*expression_identity(record), str(budget)))


def uniform_positions(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if length <= count:
        return list(range(length))
    raw = np.linspace(0, length - 1, count)
    positions = [int(round(value)) for value in raw]
    if len(set(positions)) != count:
        raise AssertionError("uniform temporal positions are not unique")
    return positions


def deterministic_video_split(
    video_ids: Iterable[str], seed: int = 42, train: int = 60, val: int = 15
) -> dict[str, list[str]]:
    values = sorted(set(str(value) for value in video_ids))
    if train <= 0 or val <= 0 or train + val >= len(values):
        raise ValueError("split sizes must leave non-empty train/val/test sets")
    rng = random.Random(seed)
    rng.shuffle(values)
    result = {
        "train": sorted(values[:train]),
        "val": sorted(values[train : train + val]),
        "test": sorted(values[train + val :]),
    }
    flattened = [item for split in result.values() for item in split]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(values):
        raise AssertionError("video split is not disjoint and exhaustive")
    return result


def sinusoidal_positions(length: int, dimension: int, device, dtype):
    import torch

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result.to(dtype=dtype)


def region_pool(patch_features, mask, grid_size: tuple[int, int]):
    """Area-weight patch pooling with an explicit empty-mask indicator."""

    import torch
    import torch.nn.functional as functional

    if patch_features.ndim != 2:
        raise ValueError("patch_features must be [patch, channel]")
    if patch_features.shape[0] != grid_size[0] * grid_size[1]:
        raise ValueError("patch count does not match grid")
    value = torch.as_tensor(mask, dtype=torch.float32, device=patch_features.device)
    if value.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    weights = functional.interpolate(
        value[None, None], size=grid_size, mode="area"
    ).reshape(-1)
    denominator = weights.sum()
    if denominator <= 0:
        return torch.zeros(
            patch_features.shape[-1],
            dtype=patch_features.dtype,
            device=patch_features.device,
        ), False
    pooled = (patch_features * weights[:, None].to(patch_features.dtype)).sum(0)
    pooled = pooled / denominator.to(patch_features.dtype)
    return pooled, True


class MatchedTrackScorer:
    """Factory namespace kept import-safe when torch is unavailable in unit tests."""

    @staticmethod
    def build(
        query_dimension: int = 256,
        visual_dimension: int = 1152,
        model_dimension: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        import torch
        import torch.nn as nn

        class _Scorer(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_projection = nn.Linear(query_dimension, model_dimension)
                self.visual_projection = nn.Linear(visual_dimension, model_dimension)
                layer = nn.TransformerEncoderLayer(
                    d_model=model_dimension,
                    nhead=heads,
                    dim_feedforward=model_dimension * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.score = nn.Sequential(
                    nn.LayerNorm(model_dimension),
                    nn.Linear(model_dimension, model_dimension),
                    nn.GELU(),
                    nn.Linear(model_dimension, 1),
                )

            def forward(self, query, tracks, temporal: bool):
                # query: [candidate, query_dimension]
                # tracks: [candidate, time, visual_dimension]
                query_token = self.query_projection(query).unsqueeze(1)
                visual_tokens = self.visual_projection(tracks)
                if temporal:
                    visual_tokens = visual_tokens + sinusoidal_positions(
                        visual_tokens.shape[1],
                        visual_tokens.shape[2],
                        visual_tokens.device,
                        visual_tokens.dtype,
                    ).unsqueeze(0)
                else:
                    visual_tokens = visual_tokens.mean(dim=1, keepdim=True)
                encoded = self.encoder(torch.cat([query_token, visual_tokens], dim=1))
                return self.score(encoded[:, 0]).squeeze(-1)

        return _Scorer()


def model_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_jsonl(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_successful_candidate_records(run_root: Path, prompt_method: str) -> dict:
    records = {}
    for shard in sorted(run_root.glob("shard-*")):
        path = shard / "candidate_records.jsonl"
        if not path.is_file():
            continue
        for record in load_jsonl([path]):
            if record.get("status") != "success":
                continue
            if record.get("prompt_method") != prompt_method:
                continue
            identity = expression_identity(record)
            if identity in records:
                raise RuntimeError(f"duplicate successful candidate identity: {identity}")
            record = dict(record)
            record["candidate_tracks_absolute_path"] = str(
                shard / record["candidate_tracks_path"]
            )
            records[identity] = record
    return records


def load_representations(run_root: Path, budget: int = 16) -> dict:
    records = {}
    path = run_root / "representation_records.jsonl"
    for record in load_jsonl([path]):
        if record.get("status") != "success" or int(record["frame_budget"]) != budget:
            continue
        identity = expression_identity(record)
        if identity in records:
            raise RuntimeError(f"duplicate representation identity: {identity}")
        value = dict(record)
        value["vector_absolute_path"] = str(run_root / record["vector_path"])
        records[identity] = value
    return records


def load_manifest_expressions(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    result = {}
    for item in manifest["objects"]:
        for expression in item["expressions"]:
            record = {
                "dataset": item["dataset"],
                "video_id": str(item["video_id"]),
                "object_id": str(item["object_id"]),
                "expression_id": str(expression["expression_id"]),
                "description_type": str(expression["type"]).lower(),
                "expression": expression["text"],
            }
            identity = expression_identity(record)
            if identity in result:
                raise RuntimeError(f"duplicate manifest expression: {identity}")
            result[identity] = record
    return manifest, result


def load_siglip_vision_model(checkpoint: Path, device: str):
    import torch
    from safetensors import safe_open
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    config_value = json.loads((checkpoint / "config.json").read_text())
    vision_value = config_value["vision_config"]["vision_tower"]
    config = SiglipVisionConfig.from_dict(vision_value)
    model = SiglipVisionModel(config)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    prefix = "model.vision_tower."
    weight_map = {
        key: filename
        for key, filename in index["weight_map"].items()
        if key.startswith(prefix)
    }
    by_file = defaultdict(list)
    for key, filename in weight_map.items():
        by_file[filename].append(key)
    state = {}
    for filename, keys in sorted(by_file.items()):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                state[key.removeprefix(prefix)] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SigLIP checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()
    processor = SiglipImageProcessor.from_pretrained(
        "/9950backfile/chenjiahui/evo_artifacts/models/"
        "siglip-so400m-patch14-384-processor"
    )
    audit = {
        "checkpoint": str(checkpoint),
        "index_sha256": sha256(checkpoint / "model.safetensors.index.json"),
        "loaded_tensors": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "hidden_size": config.hidden_size,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "selected_hidden_layer": int(vision_value["mm_vision_select_layer"]),
        "device": str(device),
        "dtype": str(dtype),
    }
    return model, processor, config, audit


def decode_rle(value: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    rle = dict(value)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


@dataclass
class FeatureRecord:
    identity: str
    dataset: str
    video_id: str
    object_id: str
    expression_id: str
    description_type: str
    expression: str
    prompt: str
    candidate_track_ids: list[int]
    candidate_confidences: list[float | None]
    selected_evaluation_positions: list[int]
    selected_frame_indices: list[int]
    selected_frame_names: list[str]
    empty_mask_observations: int
    query_vector_path: str
    candidate_tracks_path: str


def extract_features(args) -> None:
    import torch
    from PIL import Image

    torch.set_num_threads(args.cpu_threads)
    manifest_path = Path(args.manifest).resolve()
    candidate_root = Path(args.candidate_root).resolve()
    representation_root = Path(args.representation_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_expressions = load_manifest_expressions(manifest_path)
    candidates = load_successful_candidate_records(candidate_root, "concept")
    representations = load_representations(representation_root, budget=16)
    expected = set(manifest_expressions)
    if set(representations) != expected:
        raise RuntimeError(
            f"representation identity mismatch: {len(representations)} != {len(expected)}"
        )
    video_ids = sorted({identity[1] for identity in expected})
    selected_videos = {
        video_id
        for index, video_id in enumerate(video_ids)
        if index % args.num_shards == args.shard_index
    }
    selected_identities = sorted(
        identity for identity in candidates if identity[1] in selected_videos
    )
    if not selected_identities:
        raise RuntimeError("feature shard selected no successful concept records")

    model, processor, config, load_audit = load_siglip_vision_model(
        Path(args.vision_checkpoint).resolve(), args.device
    )
    grid_side = config.image_size // config.patch_size
    image_root = Path(args.image_root).resolve()
    features_by_frame = {}
    output_records = []
    feature_payloads = []
    started = time.monotonic()

    identities_by_video = defaultdict(list)
    for identity in selected_identities:
        identities_by_video[identity[1]].append(identity)
    with torch.inference_mode():
        for video_number, video_id in enumerate(sorted(identities_by_video), start=1):
            identities = identities_by_video[video_id]
            required_indices = set()
            positions_by_identity = {}
            tracks_by_identity = {}
            for identity in identities:
                candidate = candidates[identity]
                track_value = json.loads(
                    Path(candidate["candidate_tracks_absolute_path"]).read_text()
                )
                if track_value["evaluation_frame_indices"] != candidate["evaluation_frame_indices"]:
                    raise AssertionError("candidate record/track frame indices differ")
                positions = uniform_positions(
                    len(track_value["evaluation_frame_indices"]), args.temporal_steps
                )
                positions_by_identity[identity] = positions
                tracks_by_identity[identity] = track_value
                required_indices.update(
                    track_value["evaluation_frame_indices"][position]
                    for position in positions
                )
            frame_paths = sorted((image_root / video_id).glob("*.jpg"))
            if not frame_paths:
                raise FileNotFoundError(image_root / video_id)
            for offset in range(0, len(required_indices), args.image_batch_size):
                indices = sorted(required_indices)[offset : offset + args.image_batch_size]
                images = [Image.open(frame_paths[index]).convert("RGB") for index in indices]
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(device=args.device, dtype=next(model.parameters()).dtype)
                forward = model(pixels, output_hidden_states=True)
                selected = forward.hidden_states[
                    int(load_audit["selected_hidden_layer"])
                ].detach().float().cpu()
                for index, value in zip(indices, selected):
                    features_by_frame[(video_id, index)] = value
                del pixels, forward, selected

            for identity in identities:
                candidate = candidates[identity]
                track_value = tracks_by_identity[identity]
                positions = positions_by_identity[identity]
                rep = representations[identity]
                with np.load(rep["vector_absolute_path"]) as vectors:
                    query = np.asarray(vectors["z_seg"][0], dtype=np.float32)
                track_features = []
                track_ids = []
                confidences = []
                empty_count = 0
                for track in track_value["tracks"]:
                    sequence = []
                    for position in positions:
                        frame_index = track_value["evaluation_frame_indices"][position]
                        patches = features_by_frame[(video_id, frame_index)]
                        pooled, nonempty = region_pool(
                            patches,
                            decode_rle(track["frames"][position]),
                            (grid_side, grid_side),
                        )
                        if not nonempty:
                            empty_count += 1
                        sequence.append(pooled.numpy())
                    track_features.append(np.stack(sequence))
                    track_ids.append(int(track["track_id"]))
                    confidence = track.get("mean_confidence")
                    confidences.append(
                        float(confidence) if confidence is not None else None
                    )
                if track_features:
                    track_array = np.stack(track_features).astype(np.float16)
                else:
                    track_array = np.zeros(
                        (0, len(positions), config.hidden_size), dtype=np.float16
                    )
                frame_indices = [
                    track_value["evaluation_frame_indices"][position]
                    for position in positions
                ]
                metadata = manifest_expressions[identity]
                record = FeatureRecord(
                    identity="/".join(identity),
                    dataset=identity[0],
                    video_id=identity[1],
                    object_id=identity[2],
                    expression_id=identity[3],
                    description_type=metadata["description_type"],
                    expression=metadata["expression"],
                    prompt=candidate["prompt"],
                    candidate_track_ids=track_ids,
                    candidate_confidences=confidences,
                    selected_evaluation_positions=positions,
                    selected_frame_indices=frame_indices,
                    selected_frame_names=[frame_paths[index].stem for index in frame_indices],
                    empty_mask_observations=empty_count,
                    query_vector_path=rep["vector_absolute_path"],
                    candidate_tracks_path=candidate["candidate_tracks_absolute_path"],
                )
                output_records.append(asdict(record))
                feature_payloads.append(
                    {
                        "identity": record.identity,
                        "query": torch.from_numpy(query),
                        "tracks": torch.from_numpy(track_array),
                    }
                )
            for key in [key for key in features_by_frame if key[0] == video_id]:
                del features_by_frame[key]
            if video_number % 5 == 0 or video_number == len(identities_by_video):
                print(
                    f"feature videos {video_number}/{len(identities_by_video)} "
                    f"records={len(output_records)}",
                    flush=True,
                )

    torch.save(feature_payloads, output)
    metadata_path = output.with_suffix(".json")
    audit = {
        "created_at": utc_now(),
        "command": " ".join(os.sys.argv),
        "manifest": str(manifest_path),
        "candidate_root": str(candidate_root),
        "representation_root": str(representation_root),
        "candidate_generation_gt_separated": True,
        "ground_truth_read_during_feature_extraction": False,
        "prompt_method": "concept",
        "temporal_steps": args.temporal_steps,
        "cpu_threads": args.cpu_threads,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_videos": sorted(selected_videos),
        "record_count": len(output_records),
        "successful_concept_records_global": len(candidates),
        "missing_concept_records_global": len(expected - set(candidates)),
        "siglip_load_audit": load_audit,
        "feature_file": str(output),
        "feature_file_sha256": sha256(output),
        "elapsed_seconds": time.monotonic() - started,
        "records": output_records,
    }
    atomic_json(metadata_path, audit)
    print(json.dumps({key: audit[key] for key in audit if key != "records"}, indent=2))


def merge_features(args) -> None:
    import torch

    paths = [Path(value).resolve() for value in args.input]
    records = []
    metadata_records = []
    identities = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        metadata = json.loads(path.with_suffix(".json").read_text())
        by_identity = {record["identity"]: record for record in metadata["records"]}
        if len(by_identity) != len(metadata["records"]):
            raise RuntimeError(f"duplicate metadata identities in {path}")
        for value in payload:
            identity = value["identity"]
            if identity in identities:
                raise RuntimeError(f"duplicate feature identity across shards: {identity}")
            if identity not in by_identity:
                raise RuntimeError(f"missing feature metadata for {identity}")
            identities.add(identity)
            records.append(value)
            metadata_records.append(by_identity[identity])
    order = np.argsort([value["identity"] for value in records])
    records = [records[index] for index in order]
    metadata_records = [metadata_records[index] for index in order]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output)
    atomic_json(
        output.with_suffix(".json"),
        {
            "created_at": utc_now(),
            "inputs": [str(path) for path in paths],
            "input_sha256": {str(path): sha256(path) for path in paths},
            "record_count": len(records),
            "unique_identities": len(identities),
            "feature_file": str(output),
            "feature_file_sha256": sha256(output),
            "records": metadata_records,
        },
    )
    print(f"merged {len(records)} feature records into {output}")


def load_candidate_metrics(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            identity = (
                str(row["dataset"]),
                str(row["video_id"]),
                str(row["object_id"]),
                str(row["expression_id"]),
            )
            value = dict(row)
            value["candidate_metrics"] = json.loads(row["candidate_metrics"])
            result[("/".join(identity), row["prompt_method"])] = value
    return result


def top_confidence_jf(metric: dict | None) -> float:
    if not metric or not metric["candidate_metrics"]:
        return 0.0
    candidates = metric["candidate_metrics"]
    available = [value for value in candidates if value.get("confidence") is not None]
    # SAM3.1 does not expose confidence for every official tracker output.  The
    # generation order is deterministic, so use its first output only when no
    # confidence is available; neither path reads the target mask.
    selected = (
        max(available, key=lambda value: float(value["confidence"]))
        if available
        else candidates[0]
    )
    return float(selected["J_and_F"])


def candidate_values_by_track(metric: dict) -> dict[int, float]:
    return {
        int(value["track_id"]): float(value["J_and_F"])
        for value in metric["candidate_metrics"]
    }


def cluster_bootstrap_difference(rows: list[dict], seed: int, iterations: int) -> dict:
    object_values = defaultdict(list)
    object_video = {}
    for row in rows:
        if row["description_type"] != "dynamic":
            continue
        key = (row["video_id"], row["object_id"])
        object_values[key].append(
            float(row["temporal_J_and_F"]) - float(row["static_J_and_F"])
        )
        object_video[key] = row["video_id"]
    paired = {
        key: float(np.mean(values))
        for key, values in object_values.items()
        if values
    }
    by_video = defaultdict(list)
    for key, value in paired.items():
        by_video[object_video[key]].append(value)
    videos = sorted(by_video)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sampled = rng.choice(videos, len(videos), replace=True)
        values = [value for video in sampled for value in by_video[video]]
        samples.append(float(np.mean(values)))
    values = list(paired.values())
    return {
        "objects": len(values),
        "source_videos": len(videos),
        "mean_difference": float(np.mean(values)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "source_video",
    }


def aggregate_rows(rows: list[dict], method: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["object_id"], row["description_type"])].append(
            float(row[f"{method}_J_and_F"])
        )
    output = []
    for (video, object_id, description_type), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "description_type": description_type,
                "objects": 1,
                "video_id": video,
                "object_id": object_id,
                "J_and_F": float(np.mean(values)),
            }
        )
    summaries = []
    by_type = defaultdict(list)
    for row in output:
        by_type[row["description_type"]].append(row)
    for description_type, values in sorted(by_type.items()):
        summaries.append(
            {
                "method": method,
                "description_type": description_type,
                "objects": len(values),
                "source_videos": len({value["video_id"] for value in values}),
                "mean_J_and_F": float(np.mean([value["J_and_F"] for value in values])),
            }
        )
    return summaries


def train_model(
    examples: list[dict],
    validation: list[dict],
    temporal: bool,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MatchedTrackScorer.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    stale = 0
    rng = random.Random(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        losses = []
        for index in order:
            example = examples[index]
            tracks = example["tracks"].float().to(device)
            query = example["query"].float().to(device)
            query = functional.normalize(query, dim=0)
            tracks = functional.normalize(tracks, dim=-1)
            queries = query.unsqueeze(0).expand(tracks.shape[0], -1)
            logits = model(queries, tracks, temporal=temporal)
            target = torch.tensor([example["target_index"]], device=device)
            loss = functional.cross_entropy(logits.unsqueeze(0), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for example in validation:
                tracks = functional.normalize(example["tracks"].float().to(device), dim=-1)
                query = functional.normalize(example["query"].float().to(device), dim=0)
                queries = query.unsqueeze(0).expand(tracks.shape[0], -1)
                logits = model(queries, tracks, temporal=temporal)
                target = torch.tensor([example["target_index"]], device=device)
                validation_losses.append(
                    float(
                        functional.cross_entropy(logits.unsqueeze(0), target)
                        .detach()
                        .cpu()
                    )
                )
        train_loss = float(np.mean(losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best)
    return model, history, best_loss


def predict_track(model, example: dict, temporal: bool, device: str) -> int:
    import torch
    import torch.nn.functional as functional

    tracks = functional.normalize(example["tracks"].float().to(device), dim=-1)
    query = functional.normalize(example["query"].float().to(device), dim=0)
    queries = query.unsqueeze(0).expand(tracks.shape[0], -1)
    with torch.inference_mode():
        logits = model(queries, tracks, temporal=temporal)
    return int(torch.argmax(logits).item())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_and_evaluate(args) -> None:
    import torch

    feature_path = Path(args.features).resolve()
    metadata = json.loads(feature_path.with_suffix(".json").read_text())
    payload = torch.load(feature_path, map_location="cpu")
    features = {value["identity"]: value for value in payload}
    feature_metadata = {value["identity"]: value for value in metadata["records"]}
    if set(features) != set(feature_metadata):
        raise RuntimeError("feature payload and metadata identities differ")
    manifest, manifest_expressions = load_manifest_expressions(Path(args.manifest).resolve())
    split = deterministic_video_split(
        [item["video_id"] for item in manifest["objects"]],
        seed=args.split_seed,
        train=args.train_videos,
        val=args.val_videos,
    )
    split_by_video = {
        video: name for name, videos in split.items() for video in videos
    }
    metrics = load_candidate_metrics(Path(args.candidate_metrics).resolve())
    examples = []
    for identity_string, feature in features.items():
        meta = feature_metadata[identity_string]
        metric = metrics.get((identity_string, "concept"))
        if metric is None:
            raise RuntimeError(f"successful feature has no concept metric: {identity_string}")
        track_values = candidate_values_by_track(metric)
        track_ids = [int(value) for value in meta["candidate_track_ids"]]
        if len(track_ids) != feature["tracks"].shape[0]:
            raise RuntimeError(f"track feature/ID count mismatch: {identity_string}")
        oracle_track = metric["oracle_track_id"]
        oracle_jf = float(metric["oracle_J_and_F"])
        target_index = (
            track_ids.index(int(oracle_track))
            if oracle_track not in (None, "")
            else None
        )
        example = dict(feature)
        example.update(
            {
                "video_id": meta["video_id"],
                "object_id": meta["object_id"],
                "expression_id": meta["expression_id"],
                "description_type": meta["description_type"],
                "candidate_track_ids": track_ids,
                "candidate_jf_by_track": track_values,
                "oracle_J_and_F": oracle_jf,
                "target_index": target_index,
                "split": split_by_video[meta["video_id"]],
                "candidate_hit": bool(
                    target_index is not None and oracle_jf >= args.candidate_hit_threshold
                ),
            }
        )
        examples.append(example)
    training = [value for value in examples if value["split"] == "train" and value["candidate_hit"]]
    validation = [value for value in examples if value["split"] == "val" and value["candidate_hit"]]
    test_features = {value["identity"]: value for value in examples if value["split"] == "test"}
    if not training or not validation or not test_features:
        raise RuntimeError("train/validation/test feature split is empty")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed_rows = []
    prediction_rows_by_identity = defaultdict(list)
    parameter_count = None
    for seed in args.model_seeds:
        models = {}
        histories = {}
        for method, temporal in (("static", False), ("temporal", True)):
            model, history, best_loss = train_model(
                training,
                validation,
                temporal=temporal,
                seed=seed,
                device=args.device,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
            )
            count = model_parameter_count(model)
            if parameter_count is None:
                parameter_count = count
            elif parameter_count != count:
                raise AssertionError("static and temporal scorer parameter counts differ")
            models[method] = model
            histories[method] = {"best_val_loss": best_loss, "history": history}
        atomic_json(output / f"training_history_seed{seed}.json", histories)

        rows = []
        for identity, expression in sorted(manifest_expressions.items()):
            if split_by_video[identity[1]] != "test":
                continue
            identity_string = "/".join(identity)
            raw_metric = metrics.get((identity_string, "raw_expression"))
            concept_metric = metrics.get((identity_string, "concept"))
            row = {
                "seed": seed,
                "identity": identity_string,
                "dataset": identity[0],
                "video_id": identity[1],
                "object_id": identity[2],
                "expression_id": identity[3],
                "description_type": expression["description_type"],
                "expression": expression["expression"],
                "raw_expression_J_and_F": top_confidence_jf(raw_metric),
                "concept_only_J_and_F": top_confidence_jf(concept_metric),
                "concept_oracle_J_and_F": float(concept_metric["oracle_J_and_F"])
                if concept_metric
                else 0.0,
                "candidate_generation_success": int(concept_metric is not None),
                "candidate_hit": 0,
                "static_J_and_F": 0.0,
                "temporal_J_and_F": 0.0,
                "static_track_id": "",
                "temporal_track_id": "",
            }
            example = test_features.get(identity_string)
            if example is not None and example["tracks"].shape[0] > 0:
                row["candidate_hit"] = int(example["candidate_hit"])
                for method, temporal in (("static", False), ("temporal", True)):
                    selected_index = predict_track(
                        models[method], example, temporal=temporal, device=args.device
                    )
                    track_id = example["candidate_track_ids"][selected_index]
                    row[f"{method}_track_id"] = track_id
                    row[f"{method}_J_and_F"] = example["candidate_jf_by_track"].get(
                        track_id, 0.0
                    )
            rows.append(row)
            prediction_rows_by_identity[identity_string].append(row)
        write_csv(output / f"test_predictions_seed{seed}.csv", rows)
        for method in [
            "raw_expression",
            "concept_only",
            "concept_oracle",
            "static",
            "temporal",
        ]:
            for summary in aggregate_rows(rows, method):
                per_seed_rows.append({"seed": seed, **summary})

    averaged_rows = []
    for identity, values in sorted(prediction_rows_by_identity.items()):
        first = values[0]
        averaged_rows.append(
            {
                key: first[key]
                for key in [
                    "identity",
                    "dataset",
                    "video_id",
                    "object_id",
                    "expression_id",
                    "description_type",
                    "expression",
                    "candidate_generation_success",
                    "candidate_hit",
                ]
            }
            | {
                f"{method}_J_and_F": float(
                    np.mean([float(value[f"{method}_J_and_F"]) for value in values])
                )
                for method in [
                    "raw_expression",
                    "concept_only",
                    "concept_oracle",
                    "static",
                    "temporal",
                ]
            }
        )
    summaries = []
    for method in [
        "raw_expression",
        "concept_only",
        "concept_oracle",
        "static",
        "temporal",
    ]:
        summaries.extend(aggregate_rows(averaged_rows, method))
    difference = cluster_bootstrap_difference(
        averaged_rows, seed=args.split_seed, iterations=args.bootstrap_iterations
    )
    write_csv(output / "prototype_per_seed_summary.csv", per_seed_rows)
    write_csv(output / "prototype_test_predictions.csv", averaged_rows)
    write_csv(output / "prototype_summary.csv", summaries)
    atomic_json(
        output / "prototype_audit.json",
        {
            "created_at": utc_now(),
            "command": " ".join(os.sys.argv),
            "evoseg_commit": git_output(Path(args.repo).resolve(), "rev-parse", "HEAD"),
            "feature_file": str(feature_path),
            "feature_file_sha256": sha256(feature_path),
            "candidate_metrics": str(Path(args.candidate_metrics).resolve()),
            "candidate_metrics_sha256": sha256(Path(args.candidate_metrics).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256(Path(args.manifest).resolve()),
            "split": split,
            "split_unit": "source_video",
            "split_seed": args.split_seed,
            "model_seeds": args.model_seeds,
            "candidate_hit_threshold": args.candidate_hit_threshold,
            "candidate_miss_not_forced_to_wrong_positive": True,
            "confidence_selection_rule": (
                "highest available official confidence; deterministic first "
                "official output only when every confidence is unavailable"
            ),
            "train_examples": len(training),
            "validation_examples": len(validation),
            "test_expected_expressions": len(averaged_rows),
            "test_feature_records": len(test_features),
            "static_parameter_count": parameter_count,
            "temporal_parameter_count": parameter_count,
            "parameter_matched": True,
            "geometry_used_as_primary_feature": False,
            "sam31_frozen": True,
            "sa2va_frozen": True,
            "siglip_frozen": True,
            "temporal_minus_static_dynamic": difference,
        },
    )
    print(json.dumps({"summaries": summaries, "difference": difference}, indent=2))


def merge_training_predictions(args) -> None:
    """Combine independently trained seeds without retraining or selecting a seed."""

    input_paths = [Path(value).resolve() for value in args.input]
    audit_paths = [Path(value).resolve() for value in args.training_audit]
    if len(input_paths) != len(audit_paths):
        raise ValueError("each prediction input requires one training audit")
    prediction_rows = []
    seen_seed_identity = set()
    per_seed_rows = []
    seeds = set()
    methods = [
        "raw_expression",
        "concept_only",
        "concept_oracle",
        "static",
        "temporal",
    ]
    for path in input_paths:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty per-seed predictions: {path}")
        file_seeds = {int(row["seed"]) for row in rows}
        if len(file_seeds) != 1:
            raise RuntimeError(f"prediction file is not one seed: {path}")
        seed = next(iter(file_seeds))
        if seed in seeds:
            raise RuntimeError(f"duplicate model seed across inputs: {seed}")
        seeds.add(seed)
        for row in rows:
            key = (seed, row["identity"])
            if key in seen_seed_identity:
                raise RuntimeError(f"duplicate seed/identity prediction: {key}")
            seen_seed_identity.add(key)
            prediction_rows.append(row)
        for method in methods:
            for summary in aggregate_rows(rows, method):
                per_seed_rows.append({"seed": seed, **summary})

    training_audits = [json.loads(path.read_text()) for path in audit_paths]
    reference = training_audits[0]
    for audit in training_audits[1:]:
        for field in [
            "split",
            "split_seed",
            "candidate_hit_threshold",
            "static_parameter_count",
            "temporal_parameter_count",
        ]:
            if audit[field] != reference[field]:
                raise RuntimeError(f"training audit mismatch for {field}")
    if set(reference["split"]["test"]) != {
        row["video_id"] for row in prediction_rows
    }:
        raise RuntimeError("prediction videos differ from audited test split")

    grouped = defaultdict(list)
    for row in prediction_rows:
        grouped[row["identity"]].append(row)
    averaged_rows = []
    metadata_fields = [
        "identity",
        "dataset",
        "video_id",
        "object_id",
        "expression_id",
        "description_type",
        "expression",
        "candidate_generation_success",
        "candidate_hit",
    ]
    for identity, values in sorted(grouped.items()):
        if len(values) != len(seeds):
            raise RuntimeError(f"identity is missing a model seed: {identity}")
        first = values[0]
        if any(any(value[field] != first[field] for field in metadata_fields) for value in values):
            raise RuntimeError(f"metadata differs across seeds: {identity}")
        averaged_rows.append(
            {field: first[field] for field in metadata_fields}
            | {
                f"{method}_J_and_F": float(
                    np.mean([float(value[f"{method}_J_and_F"]) for value in values])
                )
                for method in methods
            }
        )
    summaries = []
    for method in methods:
        summaries.extend(aggregate_rows(averaged_rows, method))
    difference = cluster_bootstrap_difference(
        averaged_rows, seed=args.seed, iterations=args.bootstrap_iterations
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "prototype_per_seed_summary.csv", per_seed_rows)
    write_csv(output / "prototype_test_predictions.csv", averaged_rows)
    write_csv(output / "prototype_summary.csv", summaries)
    atomic_json(
        output / "prototype_audit.json",
        {
            "created_at": utc_now(),
            "command": " ".join(os.sys.argv),
            "prediction_inputs": [str(path) for path in input_paths],
            "prediction_input_sha256": {
                str(path): sha256(path) for path in input_paths
            },
            "training_audits": [str(path) for path in audit_paths],
            "training_audit_sha256": {str(path): sha256(path) for path in audit_paths},
            "model_seeds": sorted(seeds),
            "split": reference["split"],
            "split_seed": reference["split_seed"],
            "candidate_hit_threshold": reference["candidate_hit_threshold"],
            "test_expected_expressions": len(averaged_rows),
            "static_parameter_count": reference["static_parameter_count"],
            "temporal_parameter_count": reference["temporal_parameter_count"],
            "parameter_matched": True,
            "candidate_miss_not_forced_to_wrong_positive": True,
            "combines_only_test_predictions_without_seed_selection": True,
            "temporal_minus_static_dynamic": difference,
        },
    )
    print(json.dumps({"summaries": summaries, "difference": difference}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--candidate-root", required=True)
    extract.add_argument("--representation-root", required=True)
    extract.add_argument("--vision-checkpoint", required=True)
    extract.add_argument("--image-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--temporal-steps", type=int, default=8)
    extract.add_argument("--image-batch-size", type=int, default=8)
    extract.add_argument("--cpu-threads", type=int, default=1)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.set_defaults(function=extract_features)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(function=merge_features)

    train = subparsers.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--candidate-metrics", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--repo", default="/tmp/EvoSeg-temporal-compiler-sam31")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--split-seed", type=int, default=42)
    train.add_argument("--train-videos", type=int, default=60)
    train.add_argument("--val-videos", type=int, default=15)
    train.add_argument("--model-seeds", type=int, nargs="+", default=[11, 23, 42])
    train.add_argument("--candidate-hit-threshold", type=float, default=0.3)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--bootstrap-iterations", type=int, default=2000)
    train.set_defaults(function=train_and_evaluate)

    merge_training = subparsers.add_parser("merge-training")
    merge_training.add_argument("--input", action="append", required=True)
    merge_training.add_argument("--training-audit", action="append", required=True)
    merge_training.add_argument("--output-dir", required=True)
    merge_training.add_argument("--seed", type=int, default=42)
    merge_training.add_argument("--bootstrap-iterations", type=int, default=2000)
    merge_training.set_defaults(function=merge_training_predictions)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)

"""Prepare frozen T=8 track features and frozen Sa2VA query-token embeddings.

This runner never reads annotations.  Candidate masks come from the previously
frozen SAM3.1 bank; the only image features are mask-pooled frozen SigLIP
features.  Text tokens come from the frozen Sa2VA language embedding table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.temporal_compiler.temporal_matcher_prototype import (
    decode_rle,
    load_siglip_vision_model,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def frozen_embedding(model_root: Path):
    from safetensors import safe_open

    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.model.language_model.embed_tokens.weight"
    shard = model_root / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(key)
    return value, {
        "model_root": str(model_root.resolve()),
        "index_sha256": sha256(index_path),
        "embedding_shard": shard.name,
        "embedding_shape": list(value.shape),
        "embedding_dtype": str(value.dtype),
    }


def batch_region_pool(tracks: list[dict], positions: list[int], selected: list[int], frame_features: dict, grid):
    """Exactly the same area-weight pooling, batched across candidates."""
    import torch
    import torch.nn.functional as functional

    by_time=[]
    for position,frame_index in zip(positions,selected):
        masks=np.stack([decode_rle(track["frames"][position]) for track in tracks])
        weights=functional.interpolate(torch.from_numpy(masks.astype(np.float32))[:,None],size=grid,mode="area").flatten(1)
        denominator=weights.sum(-1,keepdim=True);features=frame_features[frame_index]
        pooled=weights.to(features.dtype) @ features
        pooled=pooled/denominator.clamp_min(1).to(features.dtype)
        pooled[denominator[:,0]<=0]=0
        by_time.append(pooled.numpy())
    return np.stack(by_time,axis=1)


def run(args) -> None:
    import torch
    from PIL import Image
    from transformers import AutoTokenizer

    source_path = Path(args.source_features).resolve()
    source_payload = {row["identity"]: row for row in torch.load(source_path, map_location="cpu")}
    source_meta = json.loads(source_path.with_suffix(".json").read_text())
    records = {row["identity"]: row for row in source_meta["records"]}
    manifest = json.loads(Path(args.manifest).read_text())
    object_index = {item["video_id"]: item for item in manifest["objects"]}
    expression_text = {
        "/".join((item["dataset"], item["video_id"], str(item["object_id"]), str(exp["expression_id"]))): exp["text"]
        for item in manifest["objects"] for exp in item["expressions"]
    }
    identities = sorted(source_payload)
    if args.max_expressions:
        identities = identities[: args.max_expressions]
    # Shard by source video so image encoding is never redundantly repeated
    # across workers merely because one video has many expressions.
    all_videos = sorted({records[value]["video_id"] for value in identities})
    selected_videos = {
        value for index, value in enumerate(all_videos)
        if index % args.num_shards == args.shard_index
    }
    identities = [value for value in identities if records[value]["video_id"] in selected_videos]

    model_root = Path(args.sa2va_model)
    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    embeddings, embedding_audit = frozen_embedding(model_root)

    torch.cuda.set_device(args.device)
    vision, processor, config, vision_audit = load_siglip_vision_model(
        Path(args.vision_checkpoint), f"cuda:{args.device}"
    )
    grid = config.image_size // config.patch_size
    image_root = Path(manifest["dataset"]["image_root"])
    grouped = defaultdict(list)
    for identity in identities:
        grouped[records[identity]["video_id"]].append(identity)

    output_payload = []
    output_records = []
    with torch.inference_mode():
        for video_number, (video_id, keys) in enumerate(sorted(grouped.items()), 1):
            print(f"video_start {video_number}/{len(grouped)} {video_id} expressions={len(keys)}", flush=True)
            item = object_index[video_id]
            track_payloads = {}
            required = set()
            selected_by_key = {}
            for identity in keys:
                record = records[identity]
                path = record.get("candidate_tracks_path")
                if path:
                    tracks = json.loads(Path(path).read_text())
                    positions = np.linspace(
                        0, len(tracks["evaluation_frame_indices"]) - 1,
                        num=args.track_steps, dtype=int,
                    ).tolist()
                    selected = [tracks["evaluation_frame_indices"][position] for position in positions]
                    track_payloads[identity] = tracks
                    selected_by_key[identity] = (positions, selected)
                    required.update(selected)
                else:
                    selected_by_key[identity] = ([], [])

            frame_features = {}
            ordered = sorted(required)
            for offset in range(0, len(ordered), args.image_batch_size):
                indices = ordered[offset : offset + args.image_batch_size]
                images = []
                for index in indices:
                    path = image_root / video_id / f"{item['frame_names'][index]}.jpg"
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(device=f"cuda:{args.device}", dtype=next(vision.parameters()).dtype)
                hidden = vision(pixels, output_hidden_states=True).hidden_states[
                    int(vision_audit["selected_hidden_layer"])
                ].float().cpu()
                for index, feature in zip(indices, hidden):
                    frame_features[index] = feature
            print(f"video_encoded {video_number}/{len(grouped)} {video_id} frames={len(frame_features)}", flush=True)

            pooled_track_cache = {}
            for identity in keys:
                source = source_payload[identity]
                record = records[identity]
                token_ids = tokenizer(expression_text[identity], add_special_tokens=False)["input_ids"]
                if not token_ids:
                    token_ids = [tokenizer.unk_token_id]
                token_ids = token_ids[: args.max_query_tokens]
                query_tokens = embeddings[torch.tensor(token_ids)].to(torch.float16).clone()
                candidate_features = []
                candidate_ids = []
                if identity in track_payloads:
                    tracks = track_payloads[identity]
                    positions, selected = selected_by_key[identity]
                    cache_key = (record.get("candidate_tracks_path"), tuple(positions))
                    if cache_key in pooled_track_cache:
                        candidate_features, candidate_ids = pooled_track_cache[cache_key]
                    else:
                        candidate_features = list(batch_region_pool(
                            tracks["tracks"], positions, selected, frame_features, (grid, grid)
                        ))
                        candidate_ids = [int(track["track_id"]) for track in tracks["tracks"]]
                        pooled_track_cache[cache_key] = (candidate_features, candidate_ids)
                track_array = (
                    np.stack(candidate_features).astype(np.float16)
                    if candidate_features else np.zeros((0, args.track_steps, config.hidden_size), np.float16)
                )
                output_payload.append({
                    "identity": identity,
                    "static_query": source["static_query"],
                    "temporal_states": source["temporal_states"],
                    "query_tokens": query_tokens,
                    "tracks": torch.from_numpy(track_array),
                })
                output_records.append({
                    **record,
                    "candidate_track_ids": candidate_ids,
                    "selected_frame_indices_t8": selected_by_key[identity][1],
                    "query_token_count": len(token_ids),
                })
            print(f"videos {video_number}/{len(grouped)} records={len(output_records)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output)
    output.with_suffix(".json").write_text(json.dumps({
        "command": [sys.executable, *sys.argv],
        "source_features": str(source_path),
        "source_features_sha256": sha256(source_path),
        "manifest": str(Path(args.manifest).resolve()),
        "ground_truth_read": False,
        "track_steps": args.track_steps,
        "track_sampling": "uniform over fixed candidate evaluation-frame range; real temporal order",
        "query_tokens": "frozen Sa2VA lexical token embeddings; no event phrase splitting",
        "embedding_audit": embedding_audit,
        "vision_audit": vision_audit,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sharding_unit": "source_video",
        "records": output_records,
    }, indent=2, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sa2va-model", required=True)
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--track-steps", type=int, default=8)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--max-expressions", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

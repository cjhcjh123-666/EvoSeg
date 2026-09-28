"""Extract frozen candidate-aligned features for the diagnostic probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.temporal_compiler.temporal_matcher_prototype import (
    decode_rle,
    load_siglip_vision_model,
    region_pool,
)
from projects.evoseg.temporal_grounding_mechanism.common import identity


STAGES = (1, 3, 5, 7)


def load_candidates(root: Path) -> dict[str, dict]:
    result = {}
    for shard in sorted(root.glob("shard-*")) if list(root.glob("shard-*")) else [root]:
        path = shard / "candidate_records.jsonl"
        if not path.is_file():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success" and row.get("prompt_method") == "concept":
                    key = identity(row["dataset"], row["video_id"], row["object_id"], row["expression_id"])
                    row["track_path"] = str(shard / row["candidate_tracks_path"])
                    result[key] = row
    return result


def load_states(path: Path) -> dict[str, Path]:
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "success":
                state_path = Path(row["state_path"])
                if not state_path.is_absolute():
                    state_path = path.parent / state_path
                result[row["identity"]] = state_path
    return result


def run(args) -> None:
    import torch
    from PIL import Image

    manifest = json.loads(Path(args.manifest).read_text())
    objects = manifest["objects"][: args.max_objects] if args.max_objects else manifest["objects"]
    objects = [item for index, item in enumerate(objects) if index % args.num_shards == args.shard_index]
    expression_index = {
        identity(item["dataset"], item["video_id"], item["object_id"], expression["expression_id"]): (item, expression)
        for item in objects
        for expression in item["expressions"]
    }
    candidates = load_candidates(Path(args.candidate_root))
    states = load_states(Path(args.state_records))
    missing_candidates = sorted(set(expression_index) - set(candidates))
    missing_states = sorted(set(expression_index) - set(states))
    if missing_candidates and not args.allow_missing_candidates:
        raise RuntimeError(
            f"missing candidate cache for {len(missing_candidates)} expressions; first={missing_candidates[0]}"
        )
    if missing_states and not args.allow_missing_states:
        raise RuntimeError(
            f"missing frozen state for {len(missing_states)} expressions; first={missing_states[0]}"
        )
    if missing_states:
        expression_index = {
            key: value for key, value in expression_index.items() if key not in set(missing_states)
        }
    missing_candidates = [key for key in missing_candidates if key in expression_index]
    torch.cuda.set_device(args.device)
    model, processor, config, load_audit = load_siglip_vision_model(
        Path(args.vision_checkpoint), f"cuda:{args.device}"
    )
    grid = config.image_size // config.patch_size
    image_root = Path(manifest["dataset"]["image_root"])
    by_video = defaultdict(list)
    for key, value in expression_index.items():
        if key not in set(missing_candidates):
            by_video[value[0]["video_id"]].append(key)
    payload = []
    records = []
    # A failed candidate-generation call is a genuine no-candidate outcome, not
    # a reason to silently remove the expression.  It needs no image feature:
    # both probes return no selection and candidate-direct J&F=0.  We still
    # retain the frozen query/state so the serialized schema stays identical.
    for key in missing_candidates:
        item, expression = expression_index[key]
        with np.load(states[key]) as state:
            temporal_states = np.stack(
                [np.asarray(state[f"temporal_{stage}_z"], dtype=np.float32) for stage in STAGES]
            )
            static_state = np.asarray(state["static_7_z"], dtype=np.float32)
        payload.append(
            {
                "identity": key,
                "static_query": torch.from_numpy(static_state),
                "temporal_states": torch.from_numpy(temporal_states),
                "tracks": torch.zeros((0, len(STAGES), config.hidden_size), dtype=torch.float16),
            }
        )
        records.append(
            {
                "identity": key,
                "dataset": item["dataset"],
                "split": item["split"],
                "video_id": item["video_id"],
                "object_id": item["object_id"],
                "expression_id": expression["expression_id"],
                "description_type": expression["type"],
                "expression": expression["text"],
                "candidate_track_ids": [],
                "selected_frame_indices": [],
                "candidate_tracks_path": None,
                "candidate_generation_failure": True,
                "state_path": str(states[key]),
            }
        )
    with torch.inference_mode():
        for video_number, (video_id, keys) in enumerate(sorted(by_video.items()), start=1):
            tracks = {key: json.loads(Path(candidates[key]["track_path"]).read_text()) for key in keys}
            positions = {}
            required = set()
            for key in keys:
                item, _ = expression_index[key]
                evaluation = tracks[key]["evaluation_frame_indices"]
                nominal = [int(round((stage + 1) * item["frame_count"] / 8) - 1) for stage in STAGES]
                selected = [min(range(len(evaluation)), key=lambda index: abs(evaluation[index] - endpoint)) for endpoint in nominal]
                positions[key] = selected
                required.update(evaluation[index] for index in selected)
            frame_paths = [image_root / video_id / f"{name}.jpg" for name in expression_index[keys[0]][0]["frame_names"]]
            frame_features = {}
            ordered = sorted(required)
            for offset in range(0, len(ordered), args.image_batch_size):
                indices = ordered[offset : offset + args.image_batch_size]
                images = [Image.open(frame_paths[index]).convert("RGB") for index in indices]
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(device=f"cuda:{args.device}", dtype=next(model.parameters()).dtype)
                forward = model(pixels, output_hidden_states=True)
                features = forward.hidden_states[int(load_audit["selected_hidden_layer"])].float().cpu()
                for index, value in zip(indices, features):
                    frame_features[index] = value
            for key in keys:
                item, expression = expression_index[key]
                track_value = tracks[key]
                evaluation = track_value["evaluation_frame_indices"]
                selected = positions[key]
                candidate_features = []
                track_ids = []
                for track in track_value["tracks"]:
                    sequence = []
                    for position in selected:
                        frame_index = evaluation[position]
                        pooled, _ = region_pool(
                            frame_features[frame_index], decode_rle(track["frames"][position]), (grid, grid)
                        )
                        sequence.append(pooled.numpy())
                    candidate_features.append(np.stack(sequence))
                    track_ids.append(int(track["track_id"]))
                with np.load(states[key]) as state:
                    temporal_states = np.stack([np.asarray(state[f"temporal_{stage}_z"], dtype=np.float32) for stage in STAGES])
                    static_state = np.asarray(state["static_7_z"], dtype=np.float32)
                tracks_array = (
                    np.stack(candidate_features).astype(np.float16)
                    if candidate_features
                    else np.zeros((0, len(STAGES), config.hidden_size), dtype=np.float16)
                )
                payload.append(
                    {
                        "identity": key,
                        "static_query": torch.from_numpy(static_state),
                        "temporal_states": torch.from_numpy(temporal_states),
                        "tracks": torch.from_numpy(tracks_array),
                    }
                )
                records.append(
                    {
                        "identity": key,
                        "dataset": item["dataset"],
                        "split": item["split"],
                        "video_id": item["video_id"],
                        "object_id": item["object_id"],
                        "expression_id": expression["expression_id"],
                        "description_type": expression["type"],
                        "expression": expression["text"],
                        "candidate_track_ids": track_ids,
                        "selected_frame_indices": [evaluation[index] for index in selected],
                        "candidate_tracks_path": candidates[key]["track_path"],
                        "state_path": str(states[key]),
                    }
                )
            print(f"videos {video_number}/{len(by_video)} records={len(records)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "command": [sys.executable, *sys.argv],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device),
                "ground_truth_read": False,
                "missing_candidate_identities": missing_candidates,
                "missing_candidate_count": len(missing_candidates),
                "missing_state_identities": missing_states,
                "missing_state_count": len(missing_states),
                "manifest": str(Path(args.manifest).resolve()),
                "candidate_root": str(Path(args.candidate_root).resolve()),
                "state_records": str(Path(args.state_records).resolve()),
                "vision_checkpoint": str(Path(args.vision_checkpoint).resolve()),
                "vision_load_audit": load_audit,
                "stage_indices": STAGES,
                "static_protocol": "stage-7 anchor region only plus static_7_z; anchor repeated by the scorer",
                "temporal_protocol": "four ordered candidate regions plus cumulative temporal z at stages 1/3/5/7",
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--state-records", required=True)
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--allow-missing-states", action="store_true")
    parser.add_argument("--allow-missing-candidates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

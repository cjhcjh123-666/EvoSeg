"""Build a deterministic paired-object Long-RVOS diagnostic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .sampling import nested_temporal_indices, uniform_indices


DATASET_SOURCE = "https://huggingface.co/datasets/iSEE-Laboratory/Long-RVOS"
DATASET_VERSION = "1.0"
OFFICIAL_REPO_COMMIT = "447682a4fba314e81897645a91ff1a178da493d3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_objects(videos: dict, seed: int, max_objects: int, min_videos: int):
    """Sample source videos first, then objects, without using model outputs."""
    eligible: dict[str, list[str]] = defaultdict(list)
    expression_groups = {}
    for video_id, video in videos.items():
        per_object = defaultdict(lambda: defaultdict(list))
        for expression_id, expression in video["expressions"].items():
            per_object[str(expression["obj_id"])][expression["type"]].append(
                str(expression_id)
            )
        for object_id, by_type in per_object.items():
            if by_type.get("static") and by_type.get("dynamic"):
                eligible[video_id].append(object_id)
                expression_groups[(video_id, object_id)] = by_type

    rng = random.Random(seed)
    video_ids = sorted(eligible)
    rng.shuffle(video_ids)
    for video_id in video_ids:
        rng.shuffle(eligible[video_id])

    selected = []
    # First pass guarantees video coverage before any source video contributes
    # a second object.  If 24 videos exist, the first 24 selections cover 24.
    coverage_target = min(min_videos, len(video_ids), max_objects)
    for video_id in video_ids[:coverage_target]:
        selected.append((video_id, eligible[video_id].pop()))
    # Continue one object per yet-unseen video before round-robin reuse.
    for video_id in video_ids[coverage_target:]:
        if len(selected) >= max_objects:
            break
        selected.append((video_id, eligible[video_id].pop()))
    while len(selected) < max_objects:
        added = False
        for video_id in video_ids:
            if eligible[video_id] and len(selected) < max_objects:
                selected.append((video_id, eligible[video_id].pop()))
                added = True
        if not added:
            break
    return selected, expression_groups, len(video_ids)


def build_manifest(args):
    metadata_path = Path(args.metadata).resolve()
    data = json.loads(metadata_path.read_text())
    videos = data["videos"]
    selected, groups, eligible_video_count = select_objects(
        videos, args.seed, args.max_objects, args.min_videos
    )
    image_root = Path(args.image_root).resolve()
    annotation_root = Path(args.annotation_root).resolve()

    objects = []
    missing = []
    for video_id, object_id in selected:
        video = videos[video_id]
        frame_names = [str(name) for name in video["frames"]]
        eval_indices = uniform_indices(len(frame_names), args.max_eval_frames)
        vlm_indices = nested_temporal_indices(len(frame_names), args.frame_budgets)
        expressions = []
        for expression_id, expression in video["expressions"].items():
            if str(expression["obj_id"]) != object_id:
                continue
            expressions.append(
                {
                    "expression_id": str(expression_id),
                    "type": expression["type"],
                    "text": expression["exp"],
                    "length_chars": len(expression["exp"]),
                    "length_words": len(expression["exp"].split()),
                }
            )
        expressions.sort(key=lambda item: (item["type"], item["expression_id"]))
        image_paths = [str(image_root / video_id / f"{name}.jpg") for name in frame_names]
        eval_mask_paths = [
            str(annotation_root / video_id / object_id / f"{frame_names[i]}.png")
            for i in eval_indices
        ]
        object_missing = [p for p in image_paths if not Path(p).is_file()]
        annotation_dir = annotation_root / video_id / object_id
        if not annotation_dir.is_dir():
            object_missing.append(str(annotation_dir))
        if object_missing:
            missing.extend(object_missing[:10])
        objects.append(
            {
                "dataset": "long_rvos",
                "split": "valid",
                "video_id": video_id,
                "object_id": object_id,
                "frame_names": frame_names,
                "frame_count": len(frame_names),
                "segmentation_frame_indices": eval_indices,
                "segmentation_frame_names": [frame_names[i] for i in eval_indices],
                "evaluation_frame_indices": eval_indices,
                "evaluation_frame_names": [frame_names[i] for i in eval_indices],
                "evaluation_mask_paths": eval_mask_paths,
                "evaluation_mask_present": [Path(p).is_file() for p in eval_mask_paths],
                "vlm_frame_indices": {
                    str(b): indices for b, indices in vlm_indices.items()
                },
                "vlm_frame_names": {
                    str(b): [frame_names[i] for i in indices]
                    for b, indices in vlm_indices.items()
                },
                "sam2_prompt_positions": list(range(min(args.prompt_frames, len(eval_indices)))),
                "expressions": expressions,
                "files_available": not object_missing,
            }
        )

    if args.require_files and missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"dataset extraction incomplete; examples:\n{preview}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Long-RVOS",
            "version": DATASET_VERSION,
            "split": "valid",
            "source": DATASET_SOURCE,
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "image_root": str(image_root),
            "annotation_root": str(annotation_root),
            "official_field_for_description_type": "type",
        },
        "selection": {
            "seed": args.seed,
            "method": "shuffle source videos, take one object/video, then round-robin objects",
            "max_objects": args.max_objects,
            "min_source_videos_target": args.min_videos,
            "eligible_paired_source_videos": eligible_video_count,
            "selected_objects": len(objects),
            "selected_source_videos": len({item["video_id"] for item in objects}),
            "prediction_independent": True,
        },
        "protocol": {
            "frame_budgets": args.frame_budgets,
            "experiment_a_frame_budget": 16,
            "max_evaluation_frames": args.max_eval_frames,
            "nested_sampling": "greedy farthest-point schedule with endpoints first",
            "prefix_truncation": False,
            "prompt_frames": args.prompt_frames,
            "segmentation_equals_evaluation_frames": True,
        },
        "objects": objects,
        "availability": {
            "complete": not missing,
            "missing_file_count_lower_bound": len(missing),
            "missing_examples": missing[:20],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--min-videos", type=int, default=24)
    parser.add_argument("--max-eval-frames", type=int, default=64)
    parser.add_argument("--frame-budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--prompt-frames", type=int, default=5)
    parser.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def main():
    manifest = build_manifest(parse_args())
    print(
        json.dumps(
            {
                "objects": manifest["selection"]["selected_objects"],
                "videos": manifest["selection"]["selected_source_videos"],
                "files_complete": manifest["availability"]["complete"],
            }
        )
    )


if __name__ == "__main__":
    main()

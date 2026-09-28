"""Build a prediction-independent official-train probe subset.

The subset is selected only by local file availability, official metadata and a
seed-42 video shuffle.  It never reads model predictions.  One object is taken
per source video, and all of that object's official expressions are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def uniform_indices(length: int, maximum: int = 64) -> list[int]:
    if length <= maximum:
        return list(range(length))
    values = [int(round(value)) for value in np.linspace(0, length - 1, maximum)]
    if len(set(values)) != maximum:
        raise AssertionError("uniform frame indices are not unique")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args) -> None:
    root = Path(args.dataset_root).resolve()
    metadata_path = root / "meta_expressions.json"
    metadata = json.loads(metadata_path.read_text())["videos"]
    annotation_root = root / "Annotations"
    image_root = root / "JPEGImages"
    available = sorted(
        video
        for video in metadata
        if (annotation_root / video).is_dir()
        and any((annotation_root / video).iterdir())
    )
    rng = random.Random(args.seed)
    rng.shuffle(available)
    objects = []
    selected_videos = []
    for video in available:
        value = metadata[video]
        expressions = value["expressions"]
        by_object: dict[str, list[tuple[str, dict]]] = {}
        for expression_id, expression in expressions.items():
            by_object.setdefault(str(expression["obj_id"]), []).append((str(expression_id), expression))
        eligible = [
            object_id
            for object_id, rows in sorted(by_object.items())
            if {row[1]["type"].lower() for row in rows} >= {"static", "dynamic"}
            and (annotation_root / video / object_id).is_dir()
        ]
        if not eligible:
            continue
        object_id = eligible[0]
        frame_names = [str(name) for name in value["frames"]]
        evaluation = uniform_indices(len(frame_names), 64)
        mask_paths = [annotation_root / video / object_id / f"{frame_names[index]}.png" for index in evaluation]
        if not all(path.is_file() for path in mask_paths):
            continue
        official = [
            {
                "expression_id": expression_id,
                "type": expression["type"].lower(),
                "text": expression["exp"],
                "length_chars": len(expression["exp"]),
                "length_words": len(expression["exp"].split()),
            }
            for expression_id, expression in sorted(by_object[object_id], key=lambda item: int(item[0]))
        ]
        objects.append(
            {
                "dataset": "long_rvos",
                "split": "train",
                "video_id": video,
                "object_id": object_id,
                "frame_names": frame_names,
                "frame_count": len(frame_names),
                "segmentation_frame_indices": evaluation,
                "segmentation_frame_names": [frame_names[index] for index in evaluation],
                "evaluation_frame_indices": evaluation,
                "evaluation_frame_names": [frame_names[index] for index in evaluation],
                "evaluation_mask_paths": [str(path) for path in mask_paths],
                "evaluation_mask_present": [True] * len(mask_paths),
                "expressions": official,
                "files_available": (image_root / video).is_dir(),
            }
        )
        selected_videos.append(video)
        if len(objects) == args.objects:
            break
    if len(objects) != args.objects:
        raise RuntimeError(f"only {len(objects)} eligible official-train objects available")
    overlap_path = Path(args.validation_manifest).resolve()
    validation = json.loads(overlap_path.read_text())
    overlap = set(selected_videos) & {item["video_id"] for item in validation["objects"]}
    if overlap:
        raise RuntimeError(f"train/evaluation source-video overlap: {sorted(overlap)[:5]}")
    output = {
        "schema_version": 1,
        "dataset": {
            "name": "Long-RVOS",
            "split": "official_train",
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "image_root": str(image_root),
            "annotation_root": str(annotation_root),
            "official_field_for_description_type": "type",
        },
        "selection": {
            "seed": args.seed,
            "unit": "source_video_then_first_eligible_object",
            "availability_filter": "locally extracted official annotations complete on 64 evaluation frames",
            "prediction_dependent": False,
            "objects": len(objects),
            "videos": len(selected_videos),
        },
        "objects": objects,
        "validation_manifest": str(overlap_path),
        "source_video_overlap_with_validation": [],
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"objects": len(objects), "expressions": sum(len(x["expressions"]) for x in objects), "videos": selected_videos}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

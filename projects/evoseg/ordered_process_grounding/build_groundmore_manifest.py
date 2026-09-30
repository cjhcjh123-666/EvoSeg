"""Build a Sequential-only GroundMoRe manifest from official metadata/files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(args) -> None:
    metadata_path = Path(args.metadata).resolve()
    videos = json.loads(metadata_path.read_text())["videos"]
    data_root = Path(args.data_root).resolve()
    image_root = data_root / "annotations"
    binary_root = Path(args.binary_mask_root).resolve() / args.split
    grouped = defaultdict(list)
    for video_id, video in videos.items():
        for expression_id, question in video["questions"].items():
            if question["q_type"].lower() != "sequential":
                continue
            grouped[(video_id, str(question["obj_id"]))].append({
                "expression_id": str(expression_id), "type": "sequential",
                "text": question["question"], "answer": question["answer"],
                "official_q_type": question["q_type"],
                "action_start": question["action_start"], "action_end": question["action_end"],
            })
    objects = []
    for (video_id, object_id), expressions in sorted(grouped.items()):
        directory = image_root / video_id
        frames = sorted((directory / "images").glob("frame_*.jpg"))
        if not frames:
            raise FileNotFoundError(f"official GroundMoRe frames missing: {directory}")
        frame_names = [path.stem for path in frames]
        positions = np.linspace(0, len(frames) - 1, num=min(args.evaluation_frames, len(frames)), dtype=int).tolist()
        mask_paths = []
        present = []
        object_ids = [int(value.strip()) for value in object_id.split(",")]
        for position in positions:
            source = directory / "masks" / f"{frame_names[position]}.png"
            target = binary_root / video_id / object_id.replace(",", "_") / f"{frame_names[position]}.png"
            if source.is_file():
                mask = np.asarray(Image.open(source))
                binary = np.isin(mask, object_ids).astype(np.uint8) * 255
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(binary).save(target)
                present.append(True)
            else:
                present.append(False)
            mask_paths.append(str(target))
        objects.append({
            "dataset": "groundmore", "split": args.split, "video_id": video_id,
            "object_id": object_id, "frame_count": len(frames), "frame_names": frame_names,
            "evaluation_frame_indices": positions,
            "evaluation_frame_names": [frame_names[index] for index in positions],
            "evaluation_mask_paths": mask_paths, "evaluation_mask_present": present,
            "expressions": sorted(expressions, key=lambda value: int(value["expression_id"])),
        })
    manifest = {
        "dataset": {
            "name": "GroundMoRe", "version": "v2", "split": args.split,
            "source": "https://huggingface.co/datasets/groundmore/GroundMoRe",
            "official_metadata": str(metadata_path), "official_metadata_sha256": digest(metadata_path),
            "official_code_commit": "d5074ab920a86a7ea92ecc69902109f6887f3e10",
            "image_root": str(image_root), "annotation_root": str(data_root / "annotations"),
            "binary_masks_are_evaluation_only": True,
        },
        "selection": {
            "q_type": "official Sequential only", "manual_or_llm_classification": False,
            "evaluation_frames": "20 uniformly sampled official frames (all when fewer)",
            "gt_read_during_candidate_generation": False,
        },
        "objects": objects,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "objects": len(objects), "expressions": sum(len(item["expressions"]) for item in objects),
        "videos": len({item["video_id"] for item in objects}), "output": str(output),
    }))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--binary-mask-root", required=True); parser.add_argument("--split", choices=("trainval", "test"), required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--evaluation-frames", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


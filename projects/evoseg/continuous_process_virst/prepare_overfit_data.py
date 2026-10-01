"""Build fixed training-only CPG capability subsets from existing adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(f"CPG-OVERFIT-42::{value}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(source_root: Path, output_root: Path, count: int, mode: str) -> dict:
    source_meta_path = source_root / "mevis" / "train" / "meta_expressions.json"
    source_masks_path = source_root / "mevis" / "train" / "mask_dict.json"
    source_meta = json.loads(source_meta_path.read_text())["videos"]
    source_masks = json.loads(source_masks_path.read_text())
    candidates = []
    for video_id, video in source_meta.items():
        for expression_id, expression in video["expressions"].items():
            eligible = (
                expression.get("q_type", "").lower() == "sequential"
                if mode == "groundmore"
                else bool(expression.get("explicit_order"))
            )
            if eligible:
                candidates.append((digest(f"{video_id}/{expression_id}"), video_id, expression_id))
    selected = sorted(candidates)[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} {mode} samples, found {len(selected)}")

    target = output_root / "mevis" / "train"
    image_target = target / "JPEGImages"
    image_target.mkdir(parents=True, exist_ok=True)
    videos: dict[str, dict] = {}
    annotations: set[str] = set()
    records = []
    for _, video_id, expression_id in selected:
        source_video = source_meta[video_id]
        expression = source_video["expressions"][expression_id]
        if video_id not in videos:
            videos[video_id] = {"frames": source_video["frames"], "expressions": {}}
            source_images = source_root / "mevis" / "train" / "JPEGImages" / video_id
            link = image_target / video_id
            if not link.exists():
                os.symlink(source_images.resolve(), link, target_is_directory=True)
        videos[video_id]["expressions"][expression_id] = expression
        annotations.update(expression["anno_id"])
        records.append(
            {
                "video_id": video_id,
                "expression_id": expression_id,
                "expression": expression["exp"],
                "annotation_ids": expression["anno_id"],
                "object_ids": expression.get("obj_id"),
            }
        )
    missing = sorted(annotations - source_masks.keys())
    if missing:
        raise KeyError(f"missing {len(missing)} selected annotations")
    target.mkdir(parents=True, exist_ok=True)
    (target / "meta_expressions.json").write_text(
        json.dumps({"videos": videos}, ensure_ascii=False) + "\n"
    )
    (target / "mask_dict.json").write_text(
        json.dumps({key: source_masks[key] for key in sorted(annotations)}) + "\n"
    )
    manifest = {
        "mode": mode,
        "selection": "sha256(CPG-OVERFIT-42::video_id/expression_id), ascending",
        "requested": count,
        "expressions": len(records),
        "videos": len(videos),
        "source_root": str(source_root.resolve()),
        "source_meta_sha256": file_sha256(source_meta_path),
        "records": records,
    }
    (output_root / "subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mode", choices=["groundmore", "long_rvos"], required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args.source_root, args.output_root, args.count, args.mode)))


if __name__ == "__main__":
    main()

"""Build a VIRST-compatible training view from official Long-RVOS masks."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


ORDER_PATTERN = re.compile(
    r"\b(before|after|then|first|finally|followed by|subsequently)\b", re.IGNORECASE
)


def encode_mask(mask: np.ndarray) -> dict | None:
    if not mask.any():
        return None
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def prepare(source: Path, output: Path, max_videos: int | None, explicit_only: bool) -> dict:
    metadata = json.loads((source / "meta_expressions.json").read_text())["videos"]
    image_source = source / "JPEGImages"
    annotation_source = source / "Annotations"
    target = output / "mevis" / "train"
    jpeg_target = target / "JPEGImages"
    jpeg_target.mkdir(parents=True, exist_ok=True)
    selected = {}
    mask_dict = {}
    expression_count = 0

    for video_id, video in metadata.items():
        image_dir = image_source / video_id
        annotation_dir = annotation_source / video_id
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            continue
        expressions = {
            expression_id: expression
            for expression_id, expression in video["expressions"].items()
            if not explicit_only or ORDER_PATTERN.search(expression["exp"])
        }
        if not expressions:
            continue
        if max_videos is not None and len(selected) >= max_videos:
            break
        link = jpeg_target / video_id
        if not link.exists():
            os.symlink(image_dir, link, target_is_directory=True)
        object_ids = sorted({int(value["obj_id"]) for value in expressions.values()})
        frames = [str(frame) for frame in video["frames"]]
        for object_id in object_ids:
            annotation_id = f"{video_id}__object-{object_id}"
            encoded_frames = []
            for frame in frames:
                path = annotation_dir / f"{frame}.png"
                if path.is_file():
                    with Image.open(path) as image:
                        raw = np.asarray(image.convert("P"))
                    encoded_frames.append(encode_mask(raw == object_id))
                else:
                    encoded_frames.append(None)
            mask_dict[annotation_id] = encoded_frames
        normalized_expressions = {}
        for expression_id, expression in expressions.items():
            object_id = int(expression["obj_id"])
            normalized_expressions[str(expression_id)] = {
                "exp": expression["exp"],
                "obj_id": [object_id],
                "anno_id": [f"{video_id}__object-{object_id}"],
                "type": expression.get("type"),
                "explicit_order": bool(ORDER_PATTERN.search(expression["exp"])),
            }
        selected[video_id] = {"frames": frames, "expressions": normalized_expressions}
        expression_count += len(normalized_expressions)

    target.mkdir(parents=True, exist_ok=True)
    (target / "meta_expressions.json").write_text(
        json.dumps({"videos": selected}, ensure_ascii=False) + "\n"
    )
    (target / "mask_dict.json").write_text(json.dumps(mask_dict) + "\n")
    audit = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "videos": len(selected),
        "expressions": expression_count,
        "mask_tracks": len(mask_dict),
        "explicit_only": explicit_only,
        "filter": ORDER_PATTERN.pattern,
    }
    (output / "adapter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--explicit-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.max_videos, args.explicit_only)))


if __name__ == "__main__":
    main()

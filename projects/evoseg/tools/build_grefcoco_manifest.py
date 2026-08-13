"""Build gRefCOCO (GRES) manifests for S4b training.

gRefCOCO uses COCO 2014 ``train2014`` images.  Every expression with at least
one target produces one positive record per sentence with the union of its
target masks; ``no_target`` expressions are written to a separate no-object
manifest so they can be used as anti-hallucination negatives.  The JSON split
labels are mapped to the S4b split names (testA/testB -> test).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from pycocotools import mask as coco_mask


_SPLIT_MAP = {"train": "train", "val": "val", "testA": "test", "testB": "test"}


def _rasterize_polygons(segmentation: Any, width: int, height: int) -> np.ndarray:
    if isinstance(segmentation, dict) or isinstance(segmentation, str):
        if isinstance(segmentation, str):
            segmentation = json.loads(segmentation)
        size = list(segmentation["size"])
        if isinstance(segmentation["counts"], str):
            rle = {"counts": segmentation["counts"], "size": size}
            return coco_mask.decode(rle) > 0
        rle = coco_mask.frPyObjects(
            [{"counts": list(segmentation["counts"]), "size": size}], size[0], size[1]
        )
        return coco_mask.decode(rle)[:, :, 0] > 0
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in segmentation or []:
        if not polygon:
            continue
        draw.polygon(
            [(int(polygon[i]), int(polygon[i + 1])) for i in range(0, len(polygon), 2)],
            fill=255,
        )
    return np.asarray(canvas) > 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grefs-json",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/grefs_unc.json"
        ),
    )
    parser.add_argument(
        "--instances-json",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/instances.json"
        ),
    )
    parser.add_argument(
        "--coco2014-root",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014"
        ),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/masks"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/grefcoco"
        ),
    )
    parser.add_argument("--max-grefs", type=int, default=None)
    args = parser.parse_args(argv)

    grefs = json.loads(args.grefs_json.read_text(encoding="utf-8"))
    instances = json.loads(args.instances_json.read_text(encoding="utf-8"))
    image_by_id = {int(image["id"]): image for image in instances["images"]}
    annotation_by_id = {int(annotation["id"]): annotation for annotation in instances["annotations"]}
    if args.max_grefs:
        grefs = grefs[: args.max_grefs]

    positive: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    no_object: List[Dict[str, Any]] = []
    for gref in grefs:
        image = image_by_id.get(int(gref["image_id"]))
        if image is None:
            continue
        image_id = int(gref["image_id"])
        split = _SPLIT_MAP.get(gref["split"], "train")
        image_path = args.coco2014_root / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.is_file():
            continue
        height, width = int(image["height"]), int(image["width"])
        target_annotations = [
            annotation_by_id.get(int(annotation_id))
            for annotation_id in gref["ann_id"]
        ]
        target_annotations = [annotation for annotation in target_annotations if annotation is not None]

        if gref["no_target"] or not target_annotations:
            no_object.append(
                {
                    "sample_id": f"grefcoco.no_object.{gref['ref_id']}",
                    "media_id": f"grefcoco.{image_id}",
                    "media_type": "image",
                    "media_path": str(image_path),
                    "frame_indices": [0],
                    "mask_paths": [None],
                    "target_presence": [False],
                    "anchor_position": 0,
                    "split": split,
                    "query": gref["sentences"][0]["raw"],
                    "target_id": "none",
                    "control_kind": "no_object",
                }
            )
            continue

        merged = np.zeros((height, width), dtype=bool)
        for annotation in target_annotations:
            merged |= _rasterize_polygons(annotation.get("segmentation"), width, height)
        if int(merged.sum()) == 0:
            continue
        mask_dir = args.mask_root / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{image_id}_{gref['ref_id']}.png"
        Image.fromarray((merged * 255).astype(np.uint8)).save(mask_path)
        for sentence in gref["sentences"]:
            positive[split].append(
                {
                    "sample_id": f"grefcoco.{gref['ref_id']}.{sentence['sent_id']}",
                    "media_id": f"grefcoco.{image_id}",
                    "media_type": "image",
                    "media_path": str(image_path),
                    "frame_indices": [0],
                    "mask_paths": [str(mask_path)],
                    "target_presence": [True],
                    "anchor_position": 0,
                    "split": split,
                    "query": sentence["raw"],
                    "target_id": str(gref["ref_id"]),
                    "control_kind": "positive",
                }
            )

    for split, records in positive.items():
        path = args.output / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} positive records -> {path}", flush=True)

    no_object_path = args.output / "train.no_object.jsonl"
    with no_object_path.open("w", encoding="utf-8") as handle:
        for record in no_object:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"no_object: {len(no_object)} records -> {no_object_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build category-aware no-object negatives for RefCOCO-family image manifests.

For every positive image in an existing manifest, sample foreign referring
queries whose source object category (from the RefCOCO parquet) is absent from
the target image (from COCO2017 instances annotations).  These records are
no-object controls: the query describes an object that is not present in the
image, which is exactly the shortcut the model must learn to reject.

Outputs are written under the same artifact root as the input manifest and are
meant to be consumed by train_s4b.py / the anti-shortcut evaluation protocol.
The script never touches VILA or SAM2 and never downloads anything.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from llava.evo_seg.training_contracts import MaskTrainingRecord


def _load_instances_categories(instances_path: Path) -> Dict[int, set]:
    """Return image_id -> set of COCO category ids present in the image."""
    data = json.loads(instances_path.read_text(encoding="utf-8"))
    image_categories: Dict[int, set] = defaultdict(set)
    for annotation in data["annotations"]:
        image_categories[annotation["image_id"]].add(annotation["category_id"])
    return image_categories


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_dict(record: MaskTrainingRecord) -> Dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "media_id": record.media_id,
        "media_type": record.media_type,
        "media_path": record.media_path,
        "frame_indices": list(record.frame_indices),
        "mask_paths": list(record.mask_paths),
        "target_presence": list(record.target_presence),
        "anchor_position": record.anchor_position,
        "split": record.split,
        "query": record.query,
        "target_id": record.target_id,
        "control_kind": record.control_kind,
    }


def build_image_negatives(
    variant: str,
    split_name: str,
    manifest_path: Path,
    parquet_path: Path,
    instances_paths: List[Path],
    negatives_per_image: int,
    seed: int,
    max_negatives: Optional[int],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = [r for r in _load_manifest(manifest_path) if r["control_kind"] == "positive"]
    if not records:
        raise ValueError(f"no positive records in {manifest_path}")

    # image_id -> one positive record (for media_path / media_id).
    positive_by_image: Dict[int, Dict[str, Any]] = {}
    for record in records:
        image_id = int(record["media_id"].split(".")[1])
        positive_by_image.setdefault(image_id, record)

    image_categories: Dict[int, set] = defaultdict(set)
    for instances_path in instances_paths:
        for image_id, categories in _load_instances_categories(instances_path).items():
            image_categories[image_id].update(categories)

    # Foreign query pool: (image_id, query, category_id) from the parquet.
    frame = pd.read_parquet(parquet_path)
    pool: List[Tuple[int, str, int]] = []
    for _, row in frame.iterrows():
        sentences = list(row.get("sentences", []))
        if not sentences:
            continue
        query = str(sentences[0].get("sent") or sentences[0].get("raw") or "").strip()
        if not query:
            continue
        pool.append((int(row["image_id"]), query, int(row["category_id"])))
    if not pool:
        raise ValueError(f"empty query pool from {parquet_path}")

    # Group pool queries by category for fast filtering.
    by_category: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for image_id, query, category_id in pool:
        by_category[category_id].append((image_id, query))

    negatives: List[Dict[str, Any]] = []
    target_images = sorted(positive_by_image)
    for image_id in target_images:
        present = image_categories.get(image_id, set())
        candidates: List[Tuple[int, str]] = []
        for category_id, items in by_category.items():
            if category_id in present:
                continue
            candidates.extend(items)
        if not candidates:
            continue
        rng.shuffle(candidates)
        picked = candidates[:negatives_per_image]
        target = positive_by_image[image_id]
        for source_image_id, query in picked:
            sample_id = f"{variant}.no_object.{image_id}.{source_image_id}.{len(negatives)}"
            negatives.append(
                _record_dict(
                    MaskTrainingRecord(
                        sample_id=sample_id,
                        media_id=f"coco.{image_id}",
                        media_type="image",
                        media_path=target["media_path"],
                        frame_indices=(0,),
                        mask_paths=(None,),
                        target_presence=(False,),
                        anchor_position=0,
                        split=split_name,
                        query=query,
                        target_id=None,
                        control_kind="no_object",
                    )
                )
            )
            if max_negatives is not None and len(negatives) >= max_negatives:
                return negatives
    return negatives


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="refcoco", help="refcoco/refcocoplus/refcocog")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--instances", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negatives-per-image", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    negatives = build_image_negatives(
        args.variant,
        args.split,
        args.manifest,
        args.parquet,
        args.instances,
        args.negatives_per_image,
        args.seed,
        args.max_negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in negatives:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(negatives)} no-object negatives to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

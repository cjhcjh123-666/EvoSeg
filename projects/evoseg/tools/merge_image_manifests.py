"""Merge all image segmentation sources into one v2 training manifest.

Combines the positive train records of RefCOCO/+/g (image_merged_all),
gRefCOCO, ReasonSeg and COCO-Stuff into a single ``train.jsonl``, merges the
no-object negatives, keeps the existing RefCOCO swap pairs and builds
same-image swap pairs for COCO-Stuff and gRefCOCO.  Output is written to a new
manifest dir consumed by ``train_s4b.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_pairs(records: List[Dict[str, Any]], max_pairs_per_group: int, seed: int) -> List[Dict[str, str]]:
    """Same-image different-target swap pairs (anti-shortcut)."""
    by_media: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        by_media[record["media_id"]].append(record["sample_id"])
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    for sample_ids in by_media.values():
        if len(sample_ids) < 2:
            continue
        shuffled = sample_ids[:]
        rng.shuffle(shuffled)
        count = min(max_pairs_per_group, len(shuffled) // 2)
        for index in range(count):
            pairs.append({"left_sample_id": shuffled[2 * index], "right_sample_id": shuffled[2 * index + 1]})
    return pairs


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_v2"),
    )
    parser.add_argument("--max-new-pairs", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sources",
        type=str,
        default="image_merged_all,grefcoco,reasonseg,coco_stuff",
        help="comma-separated image source dirs to merge (default: all four)",
    )
    args = parser.parse_args(argv)

    root = args.manifests_root
    sources = [part.strip() for part in args.sources.split(",") if part.strip()]
    positives: List[Dict[str, Any]] = []
    no_object: List[Dict[str, Any]] = []
    for source in sources:
        base = root / source
        train = _read_jsonl(base / "train.jsonl")
        positives.extend(train)
        no_object.extend(_read_jsonl(base / "train.no_object.jsonl"))
        print(f"{source}: {len(train)} positive, {len(_read_jsonl(base / 'train.no_object.jsonl'))} no-object", flush=True)

    existing_pairs = _read_jsonl(root / "image_merged_all" / "train.pairs.jsonl")
    new_pairs = []
    for source in sources:
        if source == "image_merged_all":
            continue
        records = _read_jsonl(root / source / "train.jsonl")
        pairs = _build_pairs(records, max_pairs_per_group=20, seed=args.seed)
        new_pairs.extend(pairs)
        print(f"{source}: {len(pairs)} new pairs", flush=True)
    rng = random.Random(args.seed)
    if len(new_pairs) > args.max_new_pairs:
        new_pairs = rng.sample(new_pairs, args.max_new_pairs)

    all_pairs = existing_pairs + new_pairs
    val = _read_jsonl(root / "image_merged_all" / "val.jsonl")

    _write_jsonl(args.output / "train.jsonl", positives)
    _write_jsonl(args.output / "train.no_object.jsonl", no_object)
    _write_jsonl(args.output / "train.pairs.jsonl", all_pairs)
    _write_jsonl(args.output / "val.jsonl", val)

    print(
        json.dumps(
            {
                "positive_train": len(positives),
                "no_object": len(no_object),
                "pairs_existing": len(existing_pairs),
                "pairs_new": len(new_pairs),
                "pairs_total": len(all_pairs),
                "val": len(val),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

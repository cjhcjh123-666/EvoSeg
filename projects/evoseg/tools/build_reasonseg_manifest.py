"""Build ReasonSeg manifests from the HF parquet packs (self-contained).

Each parquet row carries the JPEG bytes, the referring text, and the boolean
mask.  The script materialises the image and mask on disk and writes
``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` in the frozen S4b schema.
All artifacts live under ``evo_artifacts``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))


def _process_row(task: Tuple[Dict[str, Any], str, Path, Path]) -> Optional[Dict[str, Any]]:
    row, split, image_root, mask_root = task
    image_id = str(row["image_id"])
    ann_id = str(row["ann_id"])
    image_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
    raw_mask = np.asarray(row["mask"], dtype=object)
    if raw_mask.ndim == 1 and raw_mask.dtype == object and len(raw_mask) > 0 and isinstance(raw_mask[0], np.ndarray):
        mask = np.stack(list(raw_mask)).astype(bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
    if image_bytes is None or mask.size == 0 or int(mask.sum()) == 0:
        return None
    image_path = image_root / split / f"{image_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(bytes(image_bytes))
    mask_path = mask_root / split / f"{image_id}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    return {
        "sample_id": f"reasonseg.{image_id}",
        "media_id": f"reasonseg.{image_id}",
        "media_type": "image",
        "media_path": str(image_path),
        "frame_indices": [0],
        "mask_paths": [str(mask_path)],
        "target_presence": [True],
        "anchor_position": 0,
        "split": split,
        "query": str(row["text"]),
        "target_id": ann_id,
        "control_kind": "positive",
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_split(
    parquet_dir: Path,
    split: str,
    image_root: Path,
    mask_root: Path,
    output_jsonl: Path,
    max_records: Optional[int],
    workers: int,
) -> Dict[str, Any]:
    parquets = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    if not parquets:
        return {"split": split, "records": 0, "error": "no parquet files"}
    frames = [pd.read_parquet(path) for path in parquets]
    frame = pd.concat(frames, ignore_index=True)
    rows = [dict(record) for record in frame.to_dict("records")]
    if max_records:
        rows = rows[:max_records]
    tasks = [(row, split, image_root, mask_root) for row in rows]
    records: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(_process_row, tasks, chunksize=8):
            if record is not None:
                records.append(record)
    _write_jsonl(output_jsonl, records)
    return {"split": split, "rows": len(rows), "records": len(records), "output": str(output_jsonl)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/reasonseg"),
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    image_root = args.data_root / "reasonseg_images"
    mask_root = args.data_root / "reasonseg_masks"
    results = []
    for split, folder in (("train", "reasonseg_train"), ("val", "reasonseg_val"), ("test", "reasonseg_test")):
        results.append(
            build_split(
                args.data_root / folder,
                split,
                image_root,
                mask_root,
                args.output / f"{split}.jsonl",
                args.max_records,
                args.workers,
            )
        )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

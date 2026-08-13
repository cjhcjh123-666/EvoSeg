"""Build MeViS v2 training manifests for the S4b video pipeline.

MeViS v2 layout:
  train/meta_expressions_v2.json: {videos: {vid: {expressions: {eid: {exp, obj_id, anno_id}}}}}
  train/mask_dict.json:           {anno_id: [RLE per video frame, index-aligned]}
  train/JPEGImages/<vid>/<frame>.jpg

For every positive expression we emit one record per referenced object (same
query) so the model learns instance discrimination; multi-object expressions
also produce query-swap pairs between their objects.  Genuine no-target
expressions (obj_id == []) are emitted as no_object negatives.

Outputs (artifacts, outside the repo):
  .../manifests/video_mevis/{train.jsonl, train.pairs.jsonl}
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from llava.evo_seg.training_contracts import MaskTrainingManifest, MaskTrainingRecord, QuerySwapPair


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _rle_area(rle) -> int:
    try:
        return int(mask_utils.area([rle])[0])
    except Exception:
        return 0


def _build_one_video(video_id, expressions, mask_dict, jpg_root, mask_root, rng, split):
    """Return (records, pairs, n_positive, n_negative)."""
    frame_dir = jpg_root / video_id
    frame_names = sorted(
        name for name in os.listdir(frame_dir) if name.endswith(".jpg")
    )
    if len(frame_names) < 4:
        return [], [], 0, 0
    n_frames = len(frame_names)
    records: list = []
    pairs: list = []
    n_pos = n_neg = 0

    # uniform frame sampling (train/eval-aligned later in the v2 loop)
    if split == "train":
        idx = sorted(rng.sample(range(n_frames), min(6, n_frames)))
    else:
        step = max(1, n_frames // 6)
        idx = sorted(range(0, n_frames, step))[:6]
    frame_indices = tuple(int(i) for i in idx)

    def decode_present(anno_id: str, frame_index: int):
        rle_list = mask_dict.get(anno_id)
        if not rle_list or frame_index >= len(rle_list):
            return None
        rle = rle_list[frame_index]
        if rle is None:
            return None
        area = _rle_area(rle)
        if area <= 0:
            return None
        return rle

    for expr_id, info in expressions.items():
        exp = str(info.get("exp", "")).strip()
        obj_ids = [str(o) for o in info.get("obj_id", [])]
        anno_ids = [str(a) for a in info.get("anno_id", [])]
        if not exp:
            continue
        if not obj_ids or not anno_ids:
            # genuine no-target expression -> no_object negative
            records.append(
                MaskTrainingRecord(
                    sample_id=f"mevis.{split}.{video_id}.{expr_id}.notarget",
                    media_id=f"mevis.{video_id}",
                    media_type="video",
                    media_path=str(frame_dir),
                    frame_indices=frame_indices,
                    mask_paths=tuple(None for _ in frame_indices),
                    target_presence=tuple(False for _ in frame_indices),
                    anchor_position=0,
                    split=split,
                    query=exp,
                    target_id=None,
                    control_kind="no_object",
                )
            )
            n_neg += 1
            continue

        obj_records = []
        for obj_index, (obj_id, anno_id) in enumerate(zip(obj_ids, anno_ids)):
            # pick per-frame masks on the sampled frames
            mask_paths = []
            presence = []
            areas = []
            for fi in frame_indices:
                rle = decode_present(anno_id, fi)
                if rle is None:
                    mask_paths.append(None)
                    presence.append(False)
                    areas.append(0)
                    continue
                dest = mask_root / split / f"{video_id}_{anno_id}_{fi:05d}.png"
                binary = mask_utils.decode([rle])[:, :, 0]
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((binary.astype(np.uint8)) * 255).save(dest)
                mask_paths.append(str(dest))
                presence.append(True)
                areas.append(int(binary.sum()))
            # anchor = sampled frame where the target is largest
            present_idx = [i for i, a in enumerate(areas) if a > 0]
            anchor_position = present_idx[int(np.argmax([areas[i] for i in present_idx]))] if present_idx else 0
            if not any(presence):
                continue  # target never visible on the sampled frames
            record = MaskTrainingRecord(
                sample_id=f"mevis.{split}.{video_id}.{expr_id}.o{obj_index}",
                media_id=f"mevis.{video_id}",
                media_type="video",
                media_path=str(frame_dir),
                frame_indices=frame_indices,
                mask_paths=tuple(mask_paths),
                target_presence=tuple(presence),
                anchor_position=anchor_position,
                split=split,
                query=exp,
                target_id=obj_id,
                control_kind="positive",
            )
            records.append(record)
            obj_records.append(record)
            n_pos += 1
        # query-swap pairs between objects of the same multi-object expression
        for i in range(len(obj_records)):
            for j in range(i + 1, len(obj_records)):
                pairs.append(QuerySwapPair(obj_records[i].sample_id, obj_records[j].sample_id))
    return records, pairs, n_pos, n_neg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mevis-root", type=Path, default="/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2")
    parser.add_argument("--output", type=Path, default="/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_mevis")
    parser.add_argument("--mask-root", type=Path, default="/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/masks/mevis")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    meta_path = args.mevis_root / args.split / "meta_expressions_v2.json"
    mask_path = args.mevis_root / args.split / "mask_dict.json"
    jpg_root = args.mevis_root / args.split / "JPEGImages"
    meta = json.loads(meta_path.read_text())
    mask_dict = json.loads(mask_path.read_text())
    videos = meta["videos"]
    rng = random.Random(args.seed)
    all_records, all_pairs = [], []
    n_pos = n_neg = 0
    for video_id, info in videos.items():
        records, pairs, p, n = _build_one_video(
            video_id, info.get("expressions", {}), mask_dict, jpg_root, args.mask_root, rng, args.split
        )
        all_records.extend(records)
        all_pairs.extend(pairs)
        n_pos += p
        n_neg += n
    # de-dup pairs
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = tuple(sorted((pair.left_sample_id, pair.right_sample_id)))
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
    (args.output / args.split).mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "train.jsonl", all_records)
    _write_jsonl(args.output / "train.pairs.jsonl", unique_pairs)
    print(f"[mevis] split={args.split} videos={len(videos)} records={len(all_records)} "
          f"positive={n_pos} no_object={n_neg} pairs={len(unique_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

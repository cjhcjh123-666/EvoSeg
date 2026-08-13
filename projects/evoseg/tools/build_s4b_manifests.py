"""Build S4b training manifests for RefCOCO images and Ref-Youtube-VOS videos.

The script only writes external artifacts under ``evo_artifacts`` and validates
every manifest with the frozen ``MaskTrainingManifest`` contract.  It never
downloads data or modifies VILA execution paths.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from llava.evo_seg.training_contracts import (
    MaskTrainingManifest,
    MaskTrainingRecord,
    QuerySwapPair,
)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _coco_image_path(coco_root: Path, image_id: int) -> Path:
    for folder in ("train2017", "val2017"):
        candidate = coco_root / folder / f"{image_id:012d}.jpg"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"COCO2017 image missing for id {image_id}")


def _rasterize_mask(raw_anns: str, width: int, height: int, output: Path) -> None:
    annotation = json.loads(raw_anns)
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in annotation.get("segmentation", []):
        if not polygon:
            continue
        draw.polygon([(polygon[index], polygon[index + 1]) for index in range(0, len(polygon), 2)], fill=255)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _process_image_row(task) -> Tuple[int, List[MaskTrainingRecord]]:
    row, variant, split_name, coco_root, mask_root = task
    records: List[MaskTrainingRecord] = []
    image_id = int(row["image_id"])
    ann_id = int(row["ann_id"])
    image_path = _coco_image_path(coco_root, image_id)
    raw_anns = str(row["raw_anns"])
    raw_info = json.loads(str(row["raw_image_info"]))
    width, height = int(raw_info["width"]), int(raw_info["height"])
    mask_path = mask_root / variant / split_name / f"{image_id}_{ann_id}.png"
    _rasterize_mask(raw_anns, width, height, mask_path)
    for sentence_index, sentence in enumerate(row["sentences"]):
        query = str(sentence["sent"]).strip()
        if not query:
            continue
        records.append(
            MaskTrainingRecord(
                sample_id=f"{variant}.{row['ref_id']}.s{sentence_index}",
                media_id=f"coco.{image_id}",
                media_type="image",
                media_path=str(image_path),
                frame_indices=(0,),
                mask_paths=(str(mask_path),),
                target_presence=(True,),
                anchor_position=0,
                split=split_name,
                query=query,
                target_id=str(ann_id),
                control_kind="positive",
            )
        )
    return image_id, records


def build_image_manifest(
    variant: str,
    split_name: str,
    parquet_path: Path,
    coco_root: Path,
    mask_root: Path,
    max_rows: Optional[int],
    seed: int,
    workers: int,
) -> Tuple[List[MaskTrainingRecord], List[QuerySwapPair], int]:
    frame = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    groups = [group for _, group in frame.groupby("image_id")]
    rng.shuffle(groups)
    groups.sort(key=lambda group: (group["ann_id"].nunique(), len(group)), reverse=True)
    selected = []
    remaining = max_rows if max_rows is not None else len(frame)
    for group in groups:
        if remaining <= 0:
            break
        selected.append(group)
        remaining -= len(group)
    frame = pd.concat(selected, ignore_index=True)
    if max_rows is not None:
        frame = frame.head(max_rows * 2)

    tasks = [
        (row.to_dict(), variant, split_name, coco_root, mask_root)
        for _, row in frame.iterrows()
    ]
    records: List[MaskTrainingRecord] = []
    positive_by_image: Dict[int, List[MaskTrainingRecord]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, row_records in pool.map(_process_image_row, tasks, chunksize=32):
            records.extend(row_records)
            positive_by_image.setdefault(image_id, []).extend(row_records)

    pairs: List[QuerySwapPair] = []
    used_pairs = set()
    for image_id, image_records in positive_by_image.items():
        targets: Dict[str, List[MaskTrainingRecord]] = {}
        for record in image_records:
            targets.setdefault(record.target_id, []).append(record)
        target_ids = sorted(targets)
        if len(target_ids) < 2:
            continue
        left = right = None
        for left_index in range(len(target_ids)):
            for right_index in range(left_index + 1, len(target_ids)):
                for left_record in targets[target_ids[left_index]]:
                    for right_record in targets[target_ids[right_index]]:
                        if left_record.query != right_record.query:
                            left, right = left_record, right_record
                            break
                    if left is not None:
                        break
                if left is not None:
                    break
            if left is not None:
                break
        if left is None:
            continue
        key = tuple(sorted((left.sample_id, right.sample_id)))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        pairs.append(QuerySwapPair(left.sample_id, right.sample_id))

    # Negative controls on media that already has a positive record.
    sample_media = list(positive_by_image)
    if len(sample_media) >= 3:
        no_object_media = sample_media[0]
        foreign = sample_media[1]
        foreign_sentence = positive_by_image[foreign][0].query
        media_path = positive_by_image[no_object_media][0].media_path
        records.append(
            MaskTrainingRecord(
                sample_id=f"{variant}.no_object.{no_object_media}",
                media_id=f"coco.{no_object_media}",
                media_type="image",
                media_path=media_path,
                frame_indices=(0,),
                mask_paths=(None,),
                target_presence=(False,),
                anchor_position=0,
                split=split_name,
                query=f"the object described by: {foreign_sentence}",
                target_id=None,
                control_kind="no_object",
            )
        )
        records.append(
            MaskTrainingRecord(
                sample_id=f"{variant}.empty_query.{no_object_media}",
                media_id=f"coco.{no_object_media}",
                media_type="image",
                media_path=media_path,
                frame_indices=(0,),
                mask_paths=(None,),
                target_presence=(False,),
                anchor_position=0,
                split=split_name,
                query="",
                target_id=None,
                control_kind="empty_query",
            )
        )
    return records, pairs, len(frame)


def _build_pixel_to_object_map(
    ann_root: Path, video_id: str, objects: Dict[str, Dict[str, Any]]
) -> Dict[str, int]:
    """Infer per-video palette pixel value for each object from union masks."""

    frame_samples = sorted({frame for obj in objects.values() for frame in obj.get("frames", [])})[:40]
    value_frames: Dict[int, set] = {}
    for frame in frame_samples:
        mask_path = ann_root / video_id / f"{frame}.png"
        if not mask_path.is_file():
            continue
        array = np.asarray(Image.open(mask_path).convert("L"))
        for value in np.unique(array):
            if int(value) != 0:
                value_frames.setdefault(int(value), set()).add(frame)
    mapping: Dict[str, int] = {}
    for object_id, obj in objects.items():
        frames = set(obj.get("frames", [])) & set(frame_samples)
        if not frames:
            continue
        best_value, best_score = None, 0.0
        for value, value_frame_set in value_frames.items():
            score = len(value_frame_set & frames) / len(frames)
            if score > best_score:
                best_value, best_score = value, score
        if best_value is not None and best_score >= 0.8:
            mapping[object_id] = best_value
    return mapping


def _object_present_train(ann_root: Path, video_id: str, frame: int, pixel_value: int) -> bool:
    mask_path = ann_root / video_id / f"{frame:05d}.png"
    if not mask_path.is_file():
        return False
    array = np.asarray(Image.open(mask_path).convert("L"))
    return bool((array == pixel_value).any())


def _process_video(task) -> Tuple[str, List[MaskTrainingRecord], Optional[QuerySwapPair]]:
    video_id, split_name, extracted_root, mask_root = task
    extracted_root = Path(extracted_root)
    mask_root = Path(mask_root)
    if split_name == "train":
        meta_path = extracted_root / "train/meta.json"
        expr_path = extracted_root / "meta_expressions/train/meta_expressions.json"
        jpg_root = extracted_root / "train/JPEGImages"
        ann_root = extracted_root / "train/Annotations"
        meta = json.loads(meta_path.read_text())
    else:
        expr_path = extracted_root / "valid/meta_expressions_challenge.json"
        jpg_root = extracted_root / "valid/JPEGImages"
        ann_root = extracted_root / "valid/Annotations"
        meta = {
            "videos": {
                video_dir.name: {
                    "objects": {
                        object_dir.name: {
                            "frames": sorted(path.stem for path in object_dir.glob("*.png"))
                        }
                        for object_dir in video_dir.iterdir()
                        if object_dir.is_dir()
                    }
                }
                for video_dir in ann_root.iterdir()
                if video_dir.is_dir()
            }
        }
    expressions = json.loads(expr_path.read_text())
    objects = meta["videos"][video_id].get("objects", {})
    expressions_video = expressions["videos"][video_id].get("expressions", {})
    is_train = split_name == "train"
    pixel_map = _build_pixel_to_object_map(ann_root, video_id, objects) if is_train else {}

    all_frames = sorted({frame for obj in objects.values() for frame in obj.get("frames", [])})
    if len(all_frames) < 2:
        return video_id, [], None

    def presence_at(obj_id: str, frame: str) -> bool:
        if is_train:
            pixel_value = pixel_map.get(obj_id)
            return _object_present_train(ann_root, video_id, int(frame), pixel_value) if pixel_value is not None else False
        return (ann_root / video_id / obj_id / f"{int(frame):05d}.png").is_file()

    presence_counts = {
        frame: sum(presence_at(obj_id, frame) for obj_id in objects)
        for frame in all_frames
    }
    ordered = sorted(all_frames, key=lambda frame: (-presence_counts[frame], frame))
    sampled = sorted(ordered[:5])
    anchor_frame = sampled[0]

    expressions_by_obj: Dict[str, List[str]] = {}
    for info in expressions_video.values():
        obj_id = str(info["obj_id"])
        query = str(info["exp"]).strip()
        if query:
            expressions_by_obj.setdefault(obj_id, []).append(query)

    records: List[MaskTrainingRecord] = []
    video_records: List[MaskTrainingRecord] = []
    for obj_id, queries in expressions_by_obj.items():
        if obj_id not in objects:
            continue
        if not presence_at(obj_id, anchor_frame):
            continue
        if not queries:
            continue
        presence = tuple(presence_at(obj_id, frame) for frame in sampled)
        if is_train:
            pixel_value = pixel_map.get(obj_id)
            saved_paths = []
            for frame, present in zip(sampled, presence):
                if not present:
                    saved_paths.append(None)
                    continue
                source = ann_root / video_id / f"{int(frame):05d}.png"
                destination = mask_root / split_name / f"{video_id}_{obj_id}_{int(frame):05d}.png"
                if pixel_value is None:
                    raise RuntimeError(f"pixel map missing object {obj_id} in video {video_id}")
                array = np.asarray(Image.open(source).convert("L"))
                binary = ((array == pixel_value).astype(np.uint8)) * 255
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(binary).save(destination)
                saved_paths.append(str(destination))
            mask_paths = tuple(saved_paths)
        else:
            mask_paths = tuple(
                str(ann_root / video_id / obj_id / f"{int(frame):05d}.png") if present else None
                for frame, present in zip(sampled, presence)
            )
        record = MaskTrainingRecord(
            sample_id=f"ryvos.{split_name}.{video_id}.{obj_id}.{len(video_records)}",
            media_id=f"ryvos.{video_id}",
            media_type="video",
            media_path=str(jpg_root / video_id),
            frame_indices=tuple(int(frame) for frame in sampled),
            mask_paths=mask_paths,
            target_presence=presence,
            anchor_position=0,
            split=split_name,
            query=queries[0],
            target_id=obj_id,
            control_kind="positive",
        )
        records.append(record)
        video_records.append(record)

    pair = (
        QuerySwapPair(video_records[0].sample_id, video_records[1].sample_id)
        if len(video_records) >= 2
        else None
    )

    if video_records:
        absent_query = None
        for obj_id, queries in expressions_by_obj.items():
            if obj_id not in objects or obj_id in {r.target_id for r in video_records}:
                continue
            if not any(presence_at(obj_id, frame) for frame in sampled):
                absent_query = queries[0]
                break
        no_object_query = (
            f"{absent_query} (not present)" if absent_query else f"{video_records[0].query} (not present)"
        )
        records.append(
            MaskTrainingRecord(
                sample_id=f"ryvos.{split_name}.{video_id}.no_object",
                media_id=f"ryvos.{video_id}",
                media_type="video",
                media_path=str(jpg_root / video_id),
                frame_indices=tuple(int(frame) for frame in sampled),
                mask_paths=(None,) * len(sampled),
                target_presence=(False,) * len(sampled),
                anchor_position=0,
                split=split_name,
                query=no_object_query,
                target_id=None,
                control_kind="no_object",
            )
        )
        records.append(
            MaskTrainingRecord(
                sample_id=f"ryvos.{split_name}.{video_id}.empty_query",
                media_id=f"ryvos.{video_id}",
                media_type="video",
                media_path=str(jpg_root / video_id),
                frame_indices=tuple(int(frame) for frame in sampled),
                mask_paths=(None,) * len(sampled),
                target_presence=(False,) * len(sampled),
                anchor_position=0,
                split=split_name,
                query="",
                target_id=None,
                control_kind="empty_query",
            )
        )
    return video_id, records, pair


def build_video_manifest(
    extracted_root: Path,
    split_name: str,
    max_videos: Optional[int],
    seed: int,
    mask_root: Path,
    workers: int,
) -> Tuple[List[MaskTrainingRecord], List[QuerySwapPair], int]:
    if split_name == "train":
        meta_path = extracted_root / "train/meta.json"
        expr_path = extracted_root / "meta_expressions/train/meta_expressions.json"
        meta = json.loads(meta_path.read_text())
    else:
        expr_path = extracted_root / "valid/meta_expressions_challenge.json"
        meta = json.loads(expr_path.read_text())
        meta = {"videos": {video_id: {"objects": {}} for video_id in meta["videos"]}}
    expressions = json.loads(expr_path.read_text())

    rng = random.Random(seed)
    video_ids = sorted(set(meta["videos"]) & set(expressions["videos"]))
    rng.shuffle(video_ids)
    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    tasks = [(video_id, split_name, extracted_root, mask_root) for video_id in video_ids]
    records: List[MaskTrainingRecord] = []
    pairs: List[QuerySwapPair] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for _video_id, video_records, pair in pool.map(_process_video, tasks, chunksize=4):
            records.extend(video_records)
            if pair is not None:
                pairs.append(pair)
    return records, pairs, len(video_ids)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("image", "video"), required=True)
    parser.add_argument("--variant", default="refcoco", help="refcoco/refcocoplus/refcocog")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--refcoco-root", type=Path, required=True)
    parser.add_argument("--ryvos-extracted-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root
    if args.task == "image":
        for split_name, parquet_name in (("train", "train.parquet"), ("val", "validation.parquet")):
            parquet_path = args.refcoco_root / args.variant / parquet_name
            records, pairs, source_rows = build_image_manifest(
                args.variant,
                split_name,
                parquet_path,
                args.coco_root,
                artifact_root / "masks" / args.variant,
                args.max_rows,
                args.seed,
                args.workers,
            )
            manifest = MaskTrainingManifest(tuple(records), tuple(pairs))
            output_dir = artifact_root / "manifests" / f"image_{args.variant}"
            _write_jsonl(output_dir / f"{split_name}.jsonl", (_record_dict(r) for r in manifest.records))
            _write_jsonl(
                output_dir / f"{split_name}.pairs.jsonl",
                ({"left_sample_id": p.left_sample_id, "right_sample_id": p.right_sample_id} for p in manifest.query_swap_pairs),
            )
            print(f"image {args.variant} {split_name}: {len(manifest.records)} records, "
                  f"{len(manifest.query_swap_pairs)} swaps, source rows {source_rows}")
            print("  summary:", dict(manifest.summary))
    else:
        extracted_root = args.ryvos_extracted_root
        for split_name in ("train", "val"):
            records, pairs, source_videos = build_video_manifest(
                extracted_root, split_name, args.max_videos, args.seed,
                artifact_root / "masks" / "ryvos",
                args.workers,
            )
            manifest = MaskTrainingManifest(tuple(records), tuple(pairs))
            output_dir = artifact_root / "manifests" / "video_ryvos"
            _write_jsonl(output_dir / f"{split_name}.jsonl", (_record_dict(r) for r in manifest.records))
            _write_jsonl(
                output_dir / f"{split_name}.pairs.jsonl",
                ({"left_sample_id": p.left_sample_id, "right_sample_id": p.right_sample_id} for p in manifest.query_swap_pairs),
            )
            print(f"video {split_name}: {len(manifest.records)} records, "
                  f"{len(manifest.query_swap_pairs)} swaps, source videos {source_videos}")
            print("  summary:", dict(manifest.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic public Long-RVOS + MeViS-v2 pilot data without synthetic text."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from projects.evoseg.qwen_process_seg.frame_protocol import uniform_frame_indices


LONG_TYPES = ("static", "dynamic", "hybrid")


def _long_records(
    root: Path,
    per_type: int,
    frame_budget: int,
    seed: int,
) -> list[dict]:
    videos = json.loads((root / "meta_expressions.json").read_text())["videos"]
    by_type: dict[str, list[tuple[str, str]]] = {kind: [] for kind in LONG_TYPES}
    for video_id, video in videos.items():
        for expression_id, expression in video["expressions"].items():
            kind = expression.get("type", "").lower()
            if kind in by_type:
                by_type[kind].append((video_id, str(expression_id)))
    records = []
    used_videos: set[str] = set()
    for type_offset, kind in enumerate(LONG_TYPES):
        candidates = by_type[kind]
        random.Random(seed + 1009 * type_offset).shuffle(candidates)
        selected = 0
        for video_id, expression_id in candidates:
            # A video may contain multiple expressions; keep it in exactly one
            # record so the later train/validation split cannot leak frames.
            if video_id in used_videos:
                continue
            video = videos[video_id]
            expression = video["expressions"][expression_id]
            object_id = str(expression["obj_id"])
            indices = uniform_frame_indices(len(video["frames"]), frame_budget)
            names = [video["frames"][index] for index in indices]
            image_paths = [root / "JPEGImages" / video_id / f"{name}.jpg" for name in names]
            mask_paths = [
                root / "Annotations" / video_id / object_id / f"{name}.png"
                for name in names
            ]
            if not all(path.is_file() for path in image_paths):
                continue
            if not any(path.is_file() for path in mask_paths):
                continue
            records.append(
                {
                    "dataset": "long_rvos",
                    "split": "train",
                    "video_id": video_id,
                    "object_id": object_id,
                    "expression_id": expression_id,
                    "expression_type": kind,
                    "expression": expression["exp"],
                    "source_frame_indices": indices,
                    "source_frame_names": names,
                    "image_paths": [str(path) for path in image_paths],
                    # Missing official per-object PNGs are object-absent frames.
                    "mask_paths": [str(path) if path.is_file() else None for path in mask_paths],
                }
            )
            used_videos.add(video_id)
            selected += 1
            if selected == per_type:
                break
        if selected != per_type:
            raise RuntimeError(
                f"Long-RVOS requested {per_type} {kind} records, found {selected}"
            )
    random.Random(seed).shuffle(records)
    return records


def _mevis_records(
    root: Path,
    count: int,
    frame_budget: int,
    seed: int,
) -> list[dict]:
    metadata_path = root / "meta_expressions_v2.json"
    videos = json.loads(metadata_path.read_text())["videos"]
    candidates = [
        (video_id, str(expression_id))
        for video_id, video in videos.items()
        for expression_id in video["expressions"]
    ]
    random.Random(seed).shuffle(candidates)
    records = []
    used_videos = set()
    for video_id, expression_id in candidates:
        if video_id in used_videos:
            continue
        video = videos[video_id]
        expression = video["expressions"][expression_id]
        indices = uniform_frame_indices(len(video["frames"]), frame_budget)
        names = [video["frames"][index] for index in indices]
        image_paths = [root / "JPEGImages" / video_id / f"{name}.jpg" for name in names]
        if not all(path.is_file() for path in image_paths):
            continue
        annotation_ids = [str(value) for value in expression.get("anno_id", [])]
        if not annotation_ids:
            continue
        records.append(
            {
                "dataset": "mevis_v2",
                "split": "train",
                "video_id": video_id,
                "object_id": "+".join(map(str, expression.get("obj_id", []))),
                "expression_id": expression_id,
                "expression_type": "motion",
                "expression": expression["exp"],
                "source_frame_indices": indices,
                "source_frame_names": names,
                "image_paths": [str(path) for path in image_paths],
                "annotation_ids": annotation_ids,
                "mask_dictionary": str(root / "mask_dict.json"),
            }
        )
        used_videos.add(video_id)
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError(f"MeViS-v2 requested {count} records, found {len(records)}")
    return records


def build_public_video_manifest(
    long_root: Path,
    mevis_root: Path,
    long_per_type: int = 8,
    mevis_count: int = 24,
    frame_budget: int = 16,
    seed: int = 42,
    validation_fraction: float = 0.25,
) -> dict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    records = _long_records(long_root, long_per_type, frame_budget, seed)
    records.extend(
        _mevis_records(mevis_root, mevis_count, frame_budget, seed + 17)
    )
    random.Random(seed + 31).shuffle(records)

    # Stratify the held-out pilot partition by dataset/expression type. Records
    # are already unique by video within each source, so this is leakage-free.
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (record["dataset"], record["expression_type"])
        groups.setdefault(key, []).append(index)
    validation_indices: set[int] = set()
    for offset, indices in enumerate(sorted(groups.values(), key=lambda item: records[item[0]]["dataset"] + records[item[0]]["expression_type"])):
        shuffled = list(indices)
        random.Random(seed + 7919 * (offset + 1)).shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        validation_count = min(validation_count, len(shuffled) - 1)
        if validation_count <= 0:
            raise ValueError("each dataset/type group needs at least two records")
        validation_indices.update(shuffled[:validation_count])
    for index, record in enumerate(records):
        record["pilot_partition"] = (
            "validation" if index in validation_indices else "train"
        )
    return {
        "schema_version": 1,
        "name": "FTG-public-video-pilot",
        "sources": ["Long-RVOS train", "MeViS-v2 train"],
        "synthetic_expressions": False,
        "pseudo_labels": False,
        "frame_budget": frame_budget,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "partition_counts": {
            "train": len(records) - len(validation_indices),
            "validation": len(validation_indices),
        },
        "records": records,
    }


class PublicVideoPilotDataset(Dataset):
    def __init__(self, manifest: dict, partition: str | None = None) -> None:
        if partition not in (None, "train", "validation"):
            raise ValueError("partition must be train, validation, or None")
        self.records = [
            record for record in manifest["records"]
            if partition is None or record.get("pilot_partition") == partition
        ]
        self._mask_dictionaries: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _mevis_masks(self, record: dict, shape: tuple[int, int]) -> list[torch.Tensor]:
        path = record["mask_dictionary"]
        if path not in self._mask_dictionaries:
            self._mask_dictionaries[path] = json.loads(Path(path).read_text())
        dictionary = self._mask_dictionaries[path]
        result = []
        for source_index in record["source_frame_indices"]:
            merged = np.zeros(shape, dtype=bool)
            for annotation_id in record["annotation_ids"]:
                sequence = dictionary[annotation_id]
                rle = sequence[source_index] if source_index < len(sequence) else None
                if rle is not None:
                    rle = dict(rle)
                    if isinstance(rle.get("counts"), str):
                        rle["counts"] = rle["counts"].encode("ascii")
                    mask = mask_utils.decode(rle).astype(bool)
                    if mask.shape != shape:
                        mask = np.asarray(
                            Image.fromarray(mask.astype(np.uint8)).resize(
                                (shape[1], shape[0]), Image.Resampling.NEAREST
                            )
                        ).astype(bool)
                    merged |= mask
            result.append(torch.from_numpy(merged.copy()).float())
        return result

    @staticmethod
    def _long_masks(record: dict, shape: tuple[int, int]) -> list[torch.Tensor]:
        result = []
        for value in record["mask_paths"]:
            if value is None:
                mask = np.zeros(shape, dtype=bool)
            else:
                mask = np.asarray(Image.open(value).convert("L")) > 0
                if mask.shape != shape:
                    mask = np.asarray(
                        Image.fromarray(mask.astype(np.uint8)).resize(
                            (shape[1], shape[0]), Image.Resampling.NEAREST
                        )
                    ).astype(bool)
            result.append(torch.from_numpy(mask.copy()).float())
        return result

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        frames = [Image.open(path).convert("RGB") for path in record["image_paths"]]
        shape = (frames[0].height, frames[0].width)
        if record["dataset"] == "mevis_v2":
            masks = self._mevis_masks(record, shape)
        elif record["dataset"] == "long_rvos":
            masks = self._long_masks(record, shape)
        else:
            raise ValueError(f"unsupported dataset {record['dataset']!r}")
        return {**record, "frames": frames, "masks": torch.stack(masks)}

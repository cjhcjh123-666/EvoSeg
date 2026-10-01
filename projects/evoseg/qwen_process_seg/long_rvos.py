"""Deterministic Long-RVOS capability records without rewritten queries."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .frame_protocol import uniform_frame_indices


def build_capability_manifest(
    root: Path,
    count: int = 64,
    seed: int = 42,
) -> dict:
    metadata_path = root / "meta_expressions.json"
    videos = json.loads(metadata_path.read_text())["videos"]
    rng = random.Random(seed)
    video_ids = list(videos)
    rng.shuffle(video_ids)
    records = []
    for video_id in video_ids:
        video = videos[video_id]
        expression_ids = list(video["expressions"])
        rng.shuffle(expression_ids)
        for expression_id in expression_ids:
            expression = video["expressions"][expression_id]
            object_id = str(expression["obj_id"])
            indices = uniform_frame_indices(len(video["frames"]), 16)
            names = [video["frames"][index] for index in indices]
            image_paths = [root / "JPEGImages" / video_id / f"{name}.jpg" for name in names]
            mask_paths = [root / "Annotations" / video_id / object_id / f"{name}.png" for name in names]
            if all(path.is_file() for path in image_paths + mask_paths):
                records.append(
                    {
                        "dataset": "long_rvos",
                        "split": "train",
                        "video_id": video_id,
                        "object_id": object_id,
                        "expression_id": str(expression_id),
                        "expression_type": expression.get("type", "unknown"),
                        "expression": expression["exp"],
                        "source_frame_indices": indices,
                        "source_frame_names": names,
                        "image_paths": [str(path) for path in image_paths],
                        "mask_paths": [str(path) for path in mask_paths],
                    }
                )
                break
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError(f"requested {count} complete records, found {len(records)}")
    return {
        "schema_version": 1,
        "dataset": "Long-RVOS",
        "split": "train",
        "seed": seed,
        "selection": "shuffle source videos, first complete shuffled official expression",
        "frame_budget": 16,
        "records": records,
    }


class LongRVOSCapabilityDataset(Dataset):
    def __init__(self, manifest: dict) -> None:
        self.records = manifest["records"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        frames = [Image.open(path).convert("RGB") for path in record["image_paths"]]
        masks = []
        for path in record["mask_paths"]:
            mask = np.asarray(Image.open(path).convert("L")) > 0
            masks.append(torch.from_numpy(mask.copy()).float())
        return {**record, "frames": frames, "masks": torch.stack(masks)}

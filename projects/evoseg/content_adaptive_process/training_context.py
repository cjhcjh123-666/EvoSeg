"""Training-only official GroundMoRe objects and whole-process envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass
class TrainingContext:
    object_masks: Tensor | None
    target_index: Tensor | None
    process_envelope: Tensor


class GroundMoReTrainingContext:
    """Load labels after model forward; never imported by inference integration."""

    def __init__(self, dataset_root: Path, metadata: Path) -> None:
        self.dataset_root = dataset_root
        source = json.loads(metadata.read_text())["videos"]
        self.lookup = {}
        for video_id, video in source.items():
            for item in video["questions"].values():
                if item["q_type"].lower() != "sequential":
                    continue
                target_ids = tuple(int(value.strip()) for value in str(item["obj_id"]).split(","))
                normalized = " ".join(item["question"].strip().lower().split())
                self.lookup[(video_id, normalized)] = target_ids

    @staticmethod
    def question(value: str) -> str:
        for prefix in ("mevis_groundmore_train_", "mevis_cpg_groundmore_"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                break
        return " ".join(value.strip().lower().split())

    def load(self, image_path_csv: str, question: str, device: torch.device) -> TrainingContext | None:
        paths = [Path(value) for value in image_path_csv.split(",") if value]
        if not paths:
            return None
        video_id = paths[0].parent.name
        target_ids = self.lookup.get((video_id, self.question(question)))
        if target_ids is None:
            return None
        frames: list[np.ndarray | None] = []
        envelope = []
        for path in paths:
            mask_path = self.dataset_root / "annotations" / video_id / "masks" / f"{path.stem}.png"
            envelope.append(float(mask_path.is_file()))
            if mask_path.is_file():
                with Image.open(mask_path) as image:
                    frames.append(np.asarray(image.convert("P"), dtype=np.int64))
            else:
                frames.append(None)
        envelope_tensor = torch.tensor([envelope], device=device)
        shape = next((frame.shape for frame in frames if frame is not None), None)
        if shape is None or len(target_ids) != 1:
            return TrainingContext(None, None, envelope_tensor)
        dense = [frame if frame is not None else np.zeros(shape, dtype=np.int64) for frame in frames]
        ids = sorted(set(np.unique(np.stack(dense)).tolist()) - {0})
        if target_ids[0] not in ids or len(ids) < 2:
            return TrainingContext(None, None, envelope_tensor)
        masks = np.stack([np.stack([frame == object_id for frame in dense]) for object_id in ids])
        return TrainingContext(
            torch.from_numpy(masks).unsqueeze(0).to(device=device),
            torch.tensor([ids.index(target_ids[0])], dtype=torch.long, device=device),
            envelope_tensor,
        )

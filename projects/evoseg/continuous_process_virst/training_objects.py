"""Training-only GroundMoRe object trajectories for ``L_obj``.

This module is not imported by the inference wrapper.  It reads the existing
official instance PNGs for exactly the frames already selected by VIRST.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class GroundMoReTrainingObjects:
    def __init__(self, dataset_root: Path, metadata: Path) -> None:
        self.dataset_root = dataset_root
        source = json.loads(metadata.read_text())["videos"]
        self.lookup = {}
        for video_id, video in source.items():
            for item in video["questions"].values():
                if item["q_type"].lower() != "sequential":
                    continue
                target_ids = tuple(
                    int(value.strip()) for value in str(item["obj_id"]).split(",")
                )
                self.lookup[(video_id, item["question"].strip())] = target_ids

    @staticmethod
    def _question(value: str) -> str:
        prefix = "mevis_groundmore_train_"
        return value[len(prefix) :] if value.startswith(prefix) else value

    def load(
        self,
        image_path_csv: str,
        question: str,
        device: torch.device,
    ) -> tuple[Tensor, Tensor] | None:
        paths = [Path(value) for value in image_path_csv.split(",") if value]
        if not paths:
            return None
        video_id = paths[0].parent.name
        key = (video_id, self._question(question).strip())
        target_ids = self.lookup.get(key)
        if target_ids is None or len(target_ids) != 1:
            return None
        raw_frames = []
        for path in paths:
            mask_path = (
                self.dataset_root
                / "annotations"
                / video_id
                / "masks"
                / f"{path.stem}.png"
            )
            if not mask_path.is_file():
                return None
            with Image.open(mask_path) as image:
                raw_frames.append(np.asarray(image.convert("P"), dtype=np.int64))
        ids = sorted(
            set(np.unique(np.stack(raw_frames)).tolist()) - {0}
        )
        if target_ids[0] not in ids or len(ids) < 2:
            return None
        stack = np.stack(
            [np.stack([frame == object_id for frame in raw_frames]) for object_id in ids]
        )
        target_index = ids.index(target_ids[0])
        masks = torch.from_numpy(stack).unsqueeze(0).to(device=device)
        target = torch.tensor([target_index], dtype=torch.long, device=device)
        return masks, target

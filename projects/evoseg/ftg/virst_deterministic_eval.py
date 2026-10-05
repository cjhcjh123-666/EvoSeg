"""Run official VIRST evaluation with expression-invariant frame sampling."""

from __future__ import annotations

import hashlib
import os
import random
import runpy

import numpy as np


def sampling_seed(video_id: str, base_seed: int = 42) -> int:
    payload = f"evoseg-virst-ftg-v1/{base_seed}/{video_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def main() -> None:
    from data.rvos_dataset import RVOSDataset

    official_eval = os.environ.get("VIRST_OFFICIAL_EVAL_PATH", "eval.py")
    original_get_item = RVOSDataset._get_item

    def deterministic_get_item(dataset, index):
        video_id = str(dataset.d2_dataset_dicts[index]["video_name"])
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        seed = sampling_seed(video_id)
        random.seed(seed)
        np.random.seed(seed)
        try:
            return original_get_item(dataset, index)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    RVOSDataset._get_item = deterministic_get_item
    runpy.run_path(official_eval, run_name="__main__")


if __name__ == "__main__":
    main()

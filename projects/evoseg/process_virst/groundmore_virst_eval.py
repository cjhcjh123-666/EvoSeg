"""Run official VIRST on GroundMoRe with its exact 20-frame sampling rule."""

from __future__ import annotations

import os
import runpy

import numpy as np


def install_exact_20_frame_sampling() -> None:
    from data.rvos_dataset import RVOSDataset

    def sample_data_eval(dataset, index):
        record = dataset.d2_dataset_dicts[index]
        length = len(record["file_names"])
        frame_ids = np.linspace(0, length - 1, num=20, dtype=int).tolist()
        paths = [record["file_names"][position] for position in frame_ids]
        return {
            "video_name": record["video_name"],
            "video_frame_path_list": paths,
            "exp_mask_pairs": [(record["sentence"], None)],
            "length": length,
            "video_path": record["video_path"],
            "frame_ids": frame_ids,
            "exp_id": record["exp_id"],
            "ds": dataset.rvos_seg_ds_list[0],
        }

    RVOSDataset.sample_data_eval = sample_data_eval


def main() -> None:
    install_exact_20_frame_sampling()
    runpy.run_path(os.environ.get("VIRST_OFFICIAL_EVAL_PATH", "eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()

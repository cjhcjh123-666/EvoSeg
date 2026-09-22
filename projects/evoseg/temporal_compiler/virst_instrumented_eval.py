"""Run VIRST's official evaluator while auditing its realized frame sampling.

This wrapper monkey-patches only the dataset return boundary. It records the
already-sampled frame IDs and tensor lengths, then delegates to the unchanged
official ``eval.py`` through ``runpy``.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


def build_frame_audit_record(index: int, item: dict) -> dict:
    frame_ids_by_expression = item["frame_ids"]
    if not frame_ids_by_expression or frame_ids_by_expression[0] is None:
        raise AssertionError("VIRST evaluation sample did not expose frame_ids")
    first_frame_ids = [int(value) for value in frame_ids_by_expression[0]]
    if any(
        [int(value) for value in values] != first_frame_ids
        for values in frame_ids_by_expression
    ):
        raise AssertionError("expressions in one sample received different SAM frame IDs")

    vlm_frame_count = int(item["images_clip"].shape[0])
    sam_frame_count = int(item["images_sam"].shape[0])
    if vlm_frame_count != len(first_frame_ids):
        raise AssertionError(
            "official standard eval no longer uses the same realized frame count for "
            f"VLM and logged frame_ids: {vlm_frame_count} != {len(first_frame_ids)}"
        )
    if sam_frame_count != len(first_frame_ids):
        raise AssertionError(
            "official standard eval no longer uses the same realized frame count for "
            f"SAM and logged frame_ids: {sam_frame_count} != {len(first_frame_ids)}"
        )

    gt_placeholder_nonzero = 0
    for mask in item.get("masks", []):
        gt_placeholder_nonzero += int(mask.count_nonzero().item())
    if gt_placeholder_nonzero:
        raise AssertionError(
            "VIRST mevis_test model input contains a nonzero GT mask placeholder"
        )

    return {
        "dataset_index": int(index),
        "video_path": item["video_paths"][0],
        "expression_ids": [str(value) for value in item["exp_ids"]],
        "outer_sampled_frame_paths": item["image_path"].split(","),
        "model_input_frame_indices": first_frame_ids,
        "vlm_frame_count": vlm_frame_count,
        "sam_frame_count": sam_frame_count,
        "gt_placeholder_nonzero": gt_placeholder_nonzero,
        "gt_available_to_model": False,
        "frame_index_semantics": "indices_into_full_video_frame_list",
    }


def append_jsonl_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short append to {path}: {written} != {len(payload)}")
    finally:
        os.close(descriptor)


def main() -> None:
    audit_path_value = os.environ.get("VIRST_FRAME_AUDIT_PATH")
    official_eval_value = os.environ.get("VIRST_OFFICIAL_EVAL_PATH", "eval.py")
    if not audit_path_value:
        raise RuntimeError("VIRST_FRAME_AUDIT_PATH must be set")

    from data.rvos_dataset import RVOSDataset

    original_get_item = RVOSDataset._get_item

    def audited_get_item(dataset, index):
        item = original_get_item(dataset, index)
        append_jsonl_atomic(
            Path(audit_path_value), build_frame_audit_record(index, item)
        )
        return item

    RVOSDataset._get_item = audited_get_item
    runpy.run_path(official_eval_value, run_name="__main__")


if __name__ == "__main__":
    main()

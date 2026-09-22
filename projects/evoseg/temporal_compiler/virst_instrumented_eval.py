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
    sam_frame_ids_by_expression = item["frame_ids"]
    if not sam_frame_ids_by_expression or sam_frame_ids_by_expression[0] is None:
        raise AssertionError("VIRST evaluation sample did not expose frame_ids")
    sam_frame_indices = [int(value) for value in sam_frame_ids_by_expression[0]]
    if any(
        [int(value) for value in values] != sam_frame_indices
        for values in sam_frame_ids_by_expression
    ):
        raise AssertionError("expressions in one sample received different SAM frame IDs")
    vlm_frame_indices = [
        int(value) for value in item["_evoseg_vlm_original_frame_indices"]
    ]

    vlm_frame_count = int(item["images_clip"].shape[0])
    sam_frame_count = int(item["images_sam"].shape[0])
    if vlm_frame_count != len(vlm_frame_indices):
        raise AssertionError(
            "VIRST VLM tensor/frame-index count differs: "
            f"{vlm_frame_count} != {len(vlm_frame_indices)}"
        )
    if sam_frame_count != len(sam_frame_indices):
        raise AssertionError(
            "VIRST SAM tensor/frame-index count differs: "
            f"{sam_frame_count} != {len(sam_frame_indices)}"
        )
    if not set(sam_frame_indices).issubset(set(vlm_frame_indices)):
        raise AssertionError("VIRST SAM frames are not a subset of its VLM frames")

    gt_placeholder_nonzero = 0
    for mask in item.get("masks", []):
        gt_placeholder_nonzero += int(mask.count_nonzero().item())
    if gt_placeholder_nonzero:
        raise AssertionError(
            "VIRST mevis_test model input contains a nonzero GT mask placeholder"
        )

    return {
        "dataset_index": int(index),
        "video_id": Path(item["video_paths"][0]).name,
        "video_path": item["video_paths"][0],
        "output_expression_ids": [str(value) for value in item["exp_ids"]],
        "outer_sampled_frame_paths": item["image_path"].split(","),
        "model_input_frame_indices": vlm_frame_indices,
        "vlm_frame_indices": vlm_frame_indices,
        "sam_frame_indices": sam_frame_indices,
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
    completed_output_root_value = os.environ.get("VIRST_COMPLETED_OUTPUT_ROOT")
    official_eval_value = os.environ.get("VIRST_OFFICIAL_EVAL_PATH", "eval.py")
    if not audit_path_value:
        raise RuntimeError("VIRST_FRAME_AUDIT_PATH must be set")
    if not completed_output_root_value:
        raise RuntimeError("VIRST_COMPLETED_OUTPUT_ROOT must be set")

    from data.rvos_dataset import RVOSDataset

    original_get_item = RVOSDataset._get_item
    original_process_video_vlm = RVOSDataset.process_video_vlm

    def audited_process_video_vlm(dataset, video_file, data_anno, data_args):
        result = original_process_video_vlm(dataset, video_file, data_anno, data_args)
        dataset._evoseg_vlm_indices_into_outer_sample = [
            int(value) for value in result[1]
        ]
        dataset._evoseg_outer_sample_paths = [str(value) for value in video_file]
        return result

    def audited_get_item(dataset, index):
        item = original_get_item(dataset, index)
        outer_paths = [Path(value) for value in dataset._evoseg_outer_sample_paths]
        if [str(value) for value in outer_paths] != item["image_path"].split(","):
            raise AssertionError("VIRST outer frame paths changed during sample creation")
        full_video_paths = sorted(
            (
                path
                for path in Path(item["video_paths"][0]).iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=lambda path: path.name,
        )
        full_index_by_name = {path.name: position for position, path in enumerate(full_video_paths)}
        if len(full_index_by_name) != len(full_video_paths):
            raise AssertionError("duplicate frame names in VIRST source video")
        try:
            item["_evoseg_vlm_original_frame_indices"] = [
                full_index_by_name[outer_paths[position].name]
                for position in dataset._evoseg_vlm_indices_into_outer_sample
            ]
        except (IndexError, KeyError) as error:
            raise AssertionError("could not map VIRST VLM frames to full-video indices") from error
        record = build_frame_audit_record(index, item)
        item.pop("_evoseg_vlm_original_frame_indices")
        completed_root = Path(completed_output_root_value)
        completed = [
            completed_root / record["video_id"] / output_expression
            for output_expression in record["output_expression_ids"]
        ]
        if not all(path.exists() for path in completed):
            append_jsonl_atomic(Path(audit_path_value), record)
        return item

    RVOSDataset.process_video_vlm = audited_process_video_vlm
    RVOSDataset._get_item = audited_get_item
    runpy.run_path(official_eval_value, run_name="__main__")


if __name__ == "__main__":
    main()

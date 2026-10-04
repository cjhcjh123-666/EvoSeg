import json

import numpy as np
from PIL import Image

from projects.sa2va.datasets.sa2va_data_03_refvos import Sa2VA03RefVOS


def test_long_rvos_reads_official_png_masks_without_mask_dictionary(tmp_path):
    frames = tmp_path / "JPEGImages" / "video_a"
    annotations = tmp_path / "Annotations" / "video_a" / "7"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for index in range(3):
        name = f"{index:05d}"
        Image.fromarray(np.full((6, 8, 3), 50 + index, np.uint8)).save(
            frames / f"{name}.jpg"
        )
        mask = np.zeros((6, 8), np.uint8)
        mask[1:3, index:index + 2] = 255
        Image.fromarray(mask).save(annotations / f"{name}.png")
    metadata = {
        "videos": {
            "video_a": {
                "frames": ["00000", "00001", "00002"],
                "expressions": {
                    "0": {"exp": "the moving object", "obj_id": 7,
                          "type": "dynamic"}
                },
            }
        }
    }
    meta_path = tmp_path / "meta_expressions.json"
    meta_path.write_text(json.dumps(metadata))

    dataset = Sa2VA03RefVOS.__new__(Sa2VA03RefVOS)
    dataset.dataset_type = "long_rvos"
    dataset.image_folder = str(tmp_path / "JPEGImages")
    dataset.annotation_folder = str(tmp_path / "Annotations")
    videos, records, mask_dictionary = dataset.json_file_preprocess(
        str(meta_path), None
    )
    assert mask_dictionary is None
    record = records[videos["video_a"][0]]
    assert record["mask_anno_id"] == ["7"]

    mapped = dataset.dataset_map_fn([record], select_k=3)
    assert mapped["masks"].shape == (3, 6, 8)
    assert mapped["masks"].sum().item() == 12

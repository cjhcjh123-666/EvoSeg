import json

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.ftg.public_video_data import (
    PublicVideoPilotDataset,
    build_public_video_manifest,
)


def _write_image(path, value=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(path)


def _make_long(root):
    videos = {}
    for kind_index, kind in enumerate(("static", "dynamic", "hybrid")):
        for sample_index in range(2):
            video_id = f"long_{kind}_{sample_index}"
            frames = ["00000", "00001"]
            for frame in frames:
                _write_image(root / "JPEGImages" / video_id / f"{frame}.jpg")
            mask_path = root / "Annotations" / video_id / "1" / "00000.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[3:8, 4:10] = 255
            Image.fromarray(mask).save(mask_path)
            videos[video_id] = {
                "frames": frames,
                "expressions": {
                    "0": {"type": kind, "obj_id": 1, "exp": f"the {kind} object"}
                },
            }
    (root / "meta_expressions.json").write_text(json.dumps({"videos": videos}))


def _make_mevis(root):
    videos = {}
    masks = {}
    for sample_index in range(2):
        video_id = f"mevis_{sample_index}"
        frames = ["00000", "00001"]
        for frame in frames:
            _write_image(root / "JPEGImages" / video_id / f"{frame}.jpg", 120)
        annotation_id = str(sample_index + 10)
        sequence = []
        for offset in range(2):
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[2 + offset : 7 + offset, 5:11] = 1
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            sequence.append(rle)
        masks[annotation_id] = sequence
        videos[video_id] = {
            "frames": frames,
            "expressions": {
                "0": {
                    "exp": "the moving object",
                    "anno_id": [int(annotation_id)],
                    "obj_id": [sample_index + 1],
                }
            },
        }
    (root / "meta_expressions_v2.json").write_text(json.dumps({"videos": videos}))
    (root / "mask_dict.json").write_text(json.dumps(masks))


def test_manifest_is_stratified_public_and_loadable(tmp_path):
    long_root = tmp_path / "long"
    mevis_root = tmp_path / "mevis"
    long_root.mkdir()
    mevis_root.mkdir()
    _make_long(long_root)
    _make_mevis(mevis_root)
    manifest = build_public_video_manifest(
        long_root, mevis_root, long_per_type=2, mevis_count=2, frame_budget=2
    )
    assert manifest["synthetic_expressions"] is False
    assert manifest["pseudo_labels"] is False
    assert manifest["partition_counts"] == {"train": 4, "validation": 4}
    assert len({(record["dataset"], record["video_id"]) for record in manifest["records"]}) == 8

    train = PublicVideoPilotDataset(manifest, "train")
    validation = PublicVideoPilotDataset(manifest, "validation")
    assert len(train) == len(validation) == 4
    for dataset in (train, validation):
        for index in range(len(dataset)):
            sample = dataset[index]
            assert len(sample["frames"]) == 2
            assert sample["masks"].shape == (2, 12, 16)

import json
from pathlib import Path

import numpy as np
from PIL import Image

from projects.evoseg.ftg.virst_deterministic_eval import sampling_seed
from projects.evoseg.ftg.virst_long_rvos import convert, prepare


def test_sampling_seed_is_video_stable_and_video_specific():
    assert sampling_seed("video-a") == sampling_seed("video-a")
    assert sampling_seed("video-a") != sampling_seed("video-b")


def test_prepare_and_convert_fixed_long_rvos_keys(tmp_path: Path):
    source = tmp_path / "source"
    image_dir = source / "JPEGImages" / "video-a"
    image_dir.mkdir(parents=True)
    (image_dir / "00000.jpg").write_bytes(b"unused")
    (source / "meta_expressions.json").write_text(json.dumps({
        "videos": {
            "video-a": {
                "frames": ["00000"],
                "expressions": {
                    "7": {
                        "exp": "the moving object",
                        "obj_id": 3,
                        "type": "dynamic",
                    }
                },
            }
        }
    }))
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([
        {"video": "video-a", "expression_id": "7", "j": 0, "f": 0}
    ]))
    dataset = tmp_path / "dataset"
    audit = prepare(source, rows, dataset)
    assert audit["expressions"] == 1
    metadata = json.loads(
        (dataset / "mevis" / "valid" / "meta_expressions.json").read_text()
    )
    assert metadata["videos"]["video-a"]["expressions"] == {
        "expression-7": {"exp": "the moving object"}
    }

    prediction_root = tmp_path / "predictions"
    prediction_dir = prediction_root / "video-a" / "expression-7"
    prediction_dir.mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(
        prediction_dir / "00000.png"
    )
    output = tmp_path / "results.json"
    summary = convert(dataset / "expression_mapping.json", prediction_root, output)
    assert summary["expressions"] == 1
    result = json.loads(output.read_text())
    assert len(result["video-a"]["7"]["prediction_masks"]) == 1

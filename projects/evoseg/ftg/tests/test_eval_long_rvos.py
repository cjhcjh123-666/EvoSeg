import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.eval import eval_long_rvos


def _rle(mask):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode()
    return encoded


def test_long_rvos_official_temporal_metrics(tmp_path):
    video = "v1"
    expression_id = "0"
    frames = ["00000", "00005", "00010"]
    annotation_dir = tmp_path / video / "1"
    annotation_dir.mkdir(parents=True)
    target = np.zeros((4, 4), dtype=bool)
    target[1:3, 1:3] = True
    Image.fromarray(target.astype(np.uint8) * 255).save(annotation_dir / "00000.png")
    Image.fromarray(target.astype(np.uint8) * 255).save(annotation_dir / "00005.png")

    # Perfect on the first target frame, missing on the second, and a false
    # positive on the absent third frame.  Thus temporal IoU is 1/3 and vIoU
    # is also 1/3 under the official union-frame definition.
    predictions = [target, np.zeros_like(target), target]
    eval_long_rvos.WORKER_META = {
        video: {
            "frames": frames,
            "expressions": {
                expression_id: {"obj_id": 1, "type": "dynamic", "exp": "x"}
            },
        }
    }
    eval_long_rvos.WORKER_RESULTS = {
        video: {
            expression_id: {"prediction_masks": [_rle(mask) for mask in predictions]}
        }
    }
    eval_long_rvos.WORKER_ANNOTATIONS = tmp_path
    row = eval_long_rvos.evaluate_expression((video, expression_id))
    assert row["type"] == "dynamic"
    assert row["j"] == 1 / 3
    assert row["tiou"] == 1 / 3
    assert row["viou"] == 1 / 3


def test_long_rvos_summary_is_expression_averaged():
    rows = [
        {"j": 0.2, "f": 0.4, "tiou": 0.5, "viou": 0.1},
        {"j": 0.6, "f": 0.8, "tiou": 1.0, "viou": 0.3},
    ]
    summary = eval_long_rvos.summarize(rows)
    assert summary == {
        "evaluated_pairs": 2,
        "mean_j": 0.4,
        "mean_f": 0.6000000000000001,
        "j_and_f": 0.5,
        "tiou": 0.75,
        "viou": 0.2,
    }

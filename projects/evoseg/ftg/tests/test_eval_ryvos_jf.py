import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.eval import eval_ryvos_jf


def _encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return rle


def test_ref_youtube_vos_expression_uses_annotated_frames(tmp_path):
    video, expression = "video", "0"
    annotation_dir = tmp_path / video / expression
    annotation_dir.mkdir(parents=True)
    mask = np.zeros((12, 10), dtype=np.uint8)
    mask[2:8, 3:7] = 1
    Image.fromarray(mask * 255).save(annotation_dir / "00000.png")

    eval_ryvos_jf.WORKER_ANNOTATIONS = tmp_path
    eval_ryvos_jf.WORKER_RESULTS = {
        video: {
            expression: {
                "frames": ["00000", "00005"],
                "prediction_masks": [_encode(mask), _encode(np.zeros_like(mask))],
            }
        }
    }
    row = eval_ryvos_jf.evaluate_expression((video, expression))
    assert row["annotated_frames"] == 1
    assert row["j"] == 1.0
    assert row["f"] == 1.0


def test_ref_youtube_vos_summary_is_expression_averaged():
    summary = eval_ryvos_jf.summarize([
        {"j": 1.0, "f": 0.5, "annotated_frames": 1},
        {"j": 0.0, "f": 0.5, "annotated_frames": 9},
    ])
    assert summary["evaluated_pairs"] == 2
    assert summary["evaluated_annotated_frames"] == 10
    assert summary["mean_j"] == 0.5
    assert summary["mean_f"] == 0.5
    assert summary["j_and_f"] == 0.5

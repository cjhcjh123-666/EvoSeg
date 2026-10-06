"""Public hallucination metrics, independent from the historical dialog protocol."""
import numpy as np
import pytest
from PIL import Image

from projects.evoseg.hallucination.protocol import (
    COMBINATIONS, binary_union, confusion_mask_score, quartet_metrics, safe_asset)


def test_zero_mask_array_is_not_pixel_hallucination():
    union, count = binary_union([np.zeros((1, 3, 4), dtype=bool)], (3, 4))
    assert count == 1 and not union.any()
    a = np.zeros((3, 4), dtype=bool)
    b = a.copy()
    a[0, 0] = True
    b[2, 3] = True
    union, count = binary_union([a, b], (3, 4))
    assert union.sum() == 2 and count == 2
    with pytest.raises(ValueError):
        binary_union([np.zeros((2, 3))], (3, 4))
    with pytest.raises(ValueError):
        binary_union([np.full((3, 4), .7)], (3, 4))


def test_author_cms_alpha3_and_no_clamping():
    target = np.zeros((2, 2), dtype=bool)
    target[0, 0] = True
    assert confusion_mask_score(target, target) == 1.
    assert confusion_mask_score(target, np.ones((2, 2), dtype=bool)) == 2.
    other = np.zeros_like(target)
    other[1, 1] = True
    assert confusion_mask_score(target, other) == pytest.approx(1 / 3)


def test_four_combinations_prevent_always_reject_shortcut(tmp_path):
    original = np.zeros((2, 2), dtype=bool)
    original[0, 0] = True
    edited = np.ones((4, 4), dtype=bool)
    Image.fromarray(original.astype(np.uint8) * 255).save(tmp_path / 'original.png')
    Image.fromarray(edited.astype(np.uint8) * 255).save(tmp_path / 'edited.png')
    pair = {'factual_mask': str(tmp_path / 'original.png'), 'counterfactual_mask': str(tmp_path / 'edited.png')}
    outputs = {'factual_factual': original, 'factual_counterfactual': np.zeros_like(original),
               'counterfactual_factual': np.zeros_like(edited), 'counterfactual_counterfactual': edited}
    result = quartet_metrics(pair, outputs)
    assert result['factual_IoU'] == result['counterfactual_positive_IoU'] == 1.
    assert result['delta_IoU_textual'] == result['delta_IoU_visual'] == 1.
    assert result['CMS_factual'] == result['CMS_counterfactual'] == 0.
    rejected = {k: np.zeros_like(v) for k, v in outputs.items()}
    result = quartet_metrics(pair, rejected)
    assert result['CMS_factual'] == 0. and result['factual_IoU'] == 0.
    assert result['delta_IoU_textual'] == 0.
    with pytest.raises(ValueError, match='four'):
        quartet_metrics(pair, {'factual_factual': original})


def test_asset_paths_cannot_escape_dataset(tmp_path):
    (tmp_path / 'good.png').write_bytes(b'test existence only')
    assert safe_asset(tmp_path, 'good.png') == tmp_path / 'good.png'
    with pytest.raises(ValueError, match='escapes'):
        safe_asset(tmp_path, '../outside.png')

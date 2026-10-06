import numpy as np
import pytest

from projects.evoseg.dialog.protocol import RoundMetrics, assistant_history, history_reference_codes, render_query
from projects.evoseg.dialog.runtime import parse_mask_pair, token_string


def turn():
    return {'query': '[MASK-ENCODE:coco/x.jpg|refcoco|5][BOX-ENCODE:coco/x.jpg|refcoco|5] Segment the object next to instance 1.',
            'reference_rounds': [1], 'external_visual_references': []}


def no_gt(*args):
    raise AssertionError('closed-loop inference accessed GT history')


def test_closed_loop_does_not_read_gold_reference_or_assistant_targets():
    history = [{'codes': [12, 40]}]
    rendered = render_query(turn(), history, 'predicted_history', no_gt)
    assert token_string([12, 40]) in rendered
    assert 'ENCODE' not in rendered and 'instance 1' in rendered
    text, codes = assistant_history('predicted_history', token_string([12, 40]) + '<|im_end|>', [12, 40], no_gt)
    assert text == token_string([12, 40]) and codes == [12, 40]


def test_gt_and_predicted_history_are_not_silently_mixed():
    history = [{'codes': [1, 2]}]
    assert history_reference_codes(turn(), 5, history, 'gt_history', lambda _: [30, 40]) == [30, 40]
    assert history_reference_codes(turn(), 5, history, 'predicted_history', no_gt) == [1, 2]


def test_invalid_outputs_count_as_empty_predictions_not_skipped_cases():
    target = np.ones((2, 2), dtype=bool)
    metrics = RoundMetrics()
    metrics.add(2, target, np.zeros_like(target), invalid=True)
    metrics.add(2, target, target)
    score = metrics.summary()['2']
    assert score['count'] == 2 and score['invalid_outputs'] == 1
    assert score['cIoU'] == score['gIoU'] == .5


def test_native_mask_tokens_and_depth_validation():
    assert parse_mask_pair(token_string([12, 40])) == [12, 40]
    assert parse_mask_pair(token_string([12, 40]) * 2) is None
    assert parse_mask_pair('<|mt_start|><|mt_0012|><|mt_0040|><|mt_end|>') is None
    with pytest.raises(ValueError):
        token_string([256, 0])

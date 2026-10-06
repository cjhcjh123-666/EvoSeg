import inspect

import pytest

from projects.evoseg.hallucination.eval_fresh import FreshRuntime, parse_native_codes
from projects.evoseg.hallucination.public_mix import public_record


def test_native_multi_masks_and_empty_are_distinct_from_bad_codes():
    assert parse_native_codes('```json\n[]\n```') == ([], False)
    text = '<|mt_start|><|mt_0001|><|mt_0258|><|mt_end|>'
    assert parse_native_codes(text + text) == ([[1, 2], [1, 2]], False)
    assert parse_native_codes('<|mt_0001|>')[1]
    assert parse_native_codes('<|mt_0600|><|mt_0258|>')[1]


def test_inference_api_has_no_ground_truth_argument():
    assert list(inspect.signature(FreshRuntime.predict).parameters) == ['self', 'image_path', 'query']


def test_public_query_and_label_not_rewritten(tmp_path):
    (tmp_path / 'COCO_train2014_000000000072.jpg').write_bytes(b'unit existence fixture')
    ref = {'ann_id': [-1], 'no_target': True, 'ref_id': 2, 'image_id': 72, 'split': 'train'}
    sent = {'sent_id': 5, 'sent': 'giraffe on left in orange'}
    record = public_record('grefcoco', ref, sent, tmp_path, 'grefcoco')
    assert record['query'] == sent['sent'] and record['private_no_target']
    assert not record['generated_query'] and not record['pseudo_label']
    ref['no_target'] = False
    with pytest.raises(ValueError, match='disagree'):
        public_record('grefcoco', ref, sent, tmp_path, 'grefcoco')

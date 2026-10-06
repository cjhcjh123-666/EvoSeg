import copy

import pytest

from projects.evoseg.dialog.public_data import parse_conversation


def public_example():
    image = 'coco/train2014/COCO_train2014_000000376848.jpg'
    return {'conversations': [
        {'from': 'human', 'value': f'[IMAGE256:{image}] Segment a person.'},
        {'from': 'gpt', 'value': f'[MASK-DECODE:none:none:488615:{image}:refcoco].'},
        {'from': 'human', 'value': f'[MASK-ENCODE:{image}|refcoco|488615] Segment the person next to instance 1.', 'ind': 1},
        {'from': 'gpt', 'value': f'[MASK-DECODE:relational:488615:484448:{image}:refcoco].'},
    ]}


def test_public_history_and_masks_preserved_without_rewriting_annotations():
    example = public_example()
    original = copy.deepcopy(example)
    turns = parse_conversation(example)
    assert turns[0]['source_annotation_id'] == 488615
    assert turns[1]['source_annotation_id'] == 484448
    assert turns[1]['reference_rounds'] == [1]
    assert example == original


def test_unknown_history_reference_fails_closed():
    example = public_example()
    example['conversations'][2]['value'] = example['conversations'][2]['value'].replace('|488615]', '|999999]')
    with pytest.raises(ValueError, match='unobserved'):
        parse_conversation(example)


def test_invalid_pointer_is_not_silently_accepted():
    example = public_example()
    example['conversations'][2]['ind'] = 2
    with pytest.raises(ValueError, match='out of range'):
        parse_conversation(example)


def test_first_turn_external_visual_reference_is_not_fake_history():
    image = 'coco/train2014/COCO_train2014_000000376848.jpg'
    record = {'conversations': [
        {'from': 'human', 'value': f'[IMAGE256:{image}][MASK-ENCODE:{image}|refcoco|488615] Segment the other person.'},
        {'from': 'gpt', 'value': f'[MASK-DECODE:relational:488615:484448:{image}:refcoco].'}]}
    parsed = parse_conversation(record)[0]
    assert parsed['external_visual_references'] == [488615]
    assert parsed['reference_rounds'] == []

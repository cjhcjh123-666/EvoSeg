import hashlib
import json

import pytest

from projects.evoseg.dialog.pilot_data import cache_records, heldout_source_records, is_holdout
from projects.evoseg.dialog.runtime import adapter_fingerprint


def test_image_split_is_path_independent():
    for index in range(100):
        name = f'COCO_train2014_{index:012d}.jpg'
        assert is_holdout('/source/' + name) == is_holdout('/elsewhere/' + name)
    assert 0 < sum(is_holdout(f'{i}.jpg') for i in range(100)) < 100


def test_split_disjoint_and_raw_holdout(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'MANIFEST.json').write_text(json.dumps({'status': 'PUBLIC_TRAIN_PILOT_CACHE_READY',
        'generated_drafts_included': False, 'validation_images_excluded': True}))
    source = tmp_path / 'source'
    source.mkdir()
    raw = [{'query': f'RAW marker {i}'} for i in range(100)]
    path = source / 'mr_refcoco_train.json'
    path.write_text(json.dumps(raw))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (cache / 'refcoco').mkdir()
    for i in range(100):
        (cache / 'refcoco' / f'{i:06d}.json').write_text(json.dumps({
            'dataset': 'refcoco', 'image': f'/images/{i}.jpg', 'source_index': i,
            'source_sha256': digest, 'compiled_turns': [{'query': 'GT COMPILED -- NEVER EVALUATE'}]}))
    train = cache_records(cache)
    heldout = cache_records(cache, holdout=True)
    assert len(train) + len(heldout) == 100
    assert not ({r['image'] for r in train} & {r['image'] for r in heldout})
    _, selected, _ = heldout_source_records(cache, 'refcoco', source)
    assert all(r['query'].startswith('RAW') for r in selected)
    path.write_text('[]')
    with pytest.raises(RuntimeError, match='differs'):
        heldout_source_records(cache, 'refcoco', source)


def test_adapter_predictions_are_fingerprinted(tmp_path):
    assert adapter_fingerprint(None) == 'released_native'
    with pytest.raises(RuntimeError, match='missing'):
        adapter_fingerprint(tmp_path)
    weights = tmp_path / 'adapter_model.safetensors'
    weights.write_bytes(b'first adapter')
    initial = adapter_fingerprint(tmp_path)
    weights.write_bytes(b'different adapter')
    assert adapter_fingerprint(tmp_path) != initial

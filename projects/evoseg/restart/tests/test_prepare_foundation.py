import json

import pytest

from projects.evoseg.restart.prepare_foundation import atomic_json, validate_inventory


def test_complete_inventory_and_atomic_status(tmp_path):
    # Tiny synthetic files test the downloader guard, not training data.
    (tmp_path / 'model.safetensors').write_bytes(b'abc')
    atomic_json(tmp_path / 'model.safetensors.index.json', {'weight_map': {'weight': 'model.safetensors'}})
    validate_inventory(tmp_path, [{'name': 'model.safetensors', 'size': 3}])
    atomic_json(tmp_path / 'ASSETS.json', {'status': 'ASSETS_READY', 'training_started': False})
    assert json.loads((tmp_path / 'ASSETS.json').read_text())['training_started'] is False
    assert not (tmp_path / 'ASSETS.tmp').exists()


def test_partial_shard_cannot_be_ready(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'abc')
    with pytest.raises(RuntimeError, match='incomplete'):
        validate_inventory(tmp_path, [{'name': 'model.safetensors', 'size': 4}])


def test_index_must_cover_exact_published_shards(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'abc')
    atomic_json(tmp_path / 'model.safetensors.index.json', {'weight_map': {'weight': 'missing.safetensors'}})
    with pytest.raises(RuntimeError, match='index'):
        validate_inventory(tmp_path, [{'name': 'model.safetensors', 'size': 3}])


def test_empty_inventory_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match='empty'):
        validate_inventory(tmp_path, [])

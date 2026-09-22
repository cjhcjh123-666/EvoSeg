from projects.evoseg.temporal_compiler.extract_sa2va_representations import (
    select_objects,
    stable_key,
)
from projects.evoseg.temporal_compiler.analyze_sa2va_representations import distances
from projects.evoseg.temporal_compiler.merge_representation_shards import (
    merge_records,
    run as run_representation_merge,
    select_merge_inputs,
)
from projects.evoseg.temporal_compiler.instructseg_long_rvos_adapter import (
    expression_key as instructseg_expression_key,
    prepare as prepare_instructseg,
    select_objects as select_instructseg_objects,
)
from projects.evoseg.temporal_compiler.virst_long_rvos_adapter import (
    prepare as prepare_virst,
    select_objects as select_virst_objects,
)
from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    candidate_key,
    decode_rle,
    encode_rle,
    select_pilot_objects,
)
from projects.evoseg.temporal_compiler.summarize_cross_model import cluster_bootstrap

import numpy as np


def test_stable_key_keeps_expression_and_budget():
    item = {"dataset": "d", "video_id": "v", "object_id": "o"}
    expression = {"expression_id": "e"}
    assert stable_key(item, expression, 32) == "d/v/o/e/32"


def test_video_selection_keeps_all_objects_in_selected_videos():
    objects = [
        {"video_id": "a", "object_id": "1"},
        {"video_id": "a", "object_id": "2"},
        {"video_id": "b", "object_id": "1"},
    ]
    assert select_objects(objects, max_videos=1, max_objects=None) == objects[:2]


def test_representation_distance_definition():
    result = distances(np.array([2.0, 0.0]), np.array([0.0, 7.0]))
    assert np.isclose(result["cosine_similarity"], 0.0)
    assert np.isclose(result["angular_distance_radians"], np.pi / 2)
    assert np.isclose(result["normalized_l2"], np.sqrt(2))


def test_candidate_protocol_identity_and_selection():
    item = {"dataset": "d", "video_id": "v", "object_id": "o"}
    expression = {"expression_id": "e"}
    assert candidate_key(item, expression, "concept") == "d/v/o/e/concept"
    values = [{"object_id": str(index)} for index in range(40)]
    assert select_pilot_objects(values, 32) == values[:32]


def test_candidate_rle_round_trip():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))


def test_cluster_bootstrap_preserves_constant_gap():
    rows = [
        {"video_id": "a", "dynamic_minus_static_J_and_F": -0.1},
        {"video_id": "b", "dynamic_minus_static_J_and_F": -0.1},
    ]
    point, low, high = cluster_bootstrap(rows, iterations=20, seed=42)
    assert np.isclose(point, -0.1)
    assert np.isclose(low, -0.1)
    assert np.isclose(high, -0.1)


def test_merge_representation_shards_is_unique_and_ordered(tmp_path):
    base = tmp_path / "representation_records.jsonl"
    shard = tmp_path / "representation_records.worker-00-of-02.jsonl"
    base.write_text('{"key":"a","status":"success","vector_sha256":"1"}\n')
    shard.write_text('{"key":"b","status":"success","vector_sha256":"2"}\n')
    rows, audit = merge_records([base, shard])
    assert [row["key"] for row in rows] == ["a", "b"]
    assert audit["success_rows"] == 2
    assert audit["duplicate_rows"] == 0


def test_representation_merge_rerun_does_not_count_merged_output(tmp_path):
    output = tmp_path / "representation_records.jsonl"
    shard = tmp_path / "representation_records.worker-00-of-02.jsonl"
    output.write_text('{"key":"a","status":"success","vector_sha256":"1"}\n')
    shard.write_text('{"key":"b","status":"success","vector_sha256":"2"}\n')
    args = type("Args", (), {"run_dir": str(tmp_path), "expected_count": 2})()
    assert select_merge_inputs(output, [shard], 2) == [output, shard]
    assert run_representation_merge(args) == 0
    assert select_merge_inputs(output, [shard], 2) == [output]
    assert run_representation_merge(args) == 0
    audit = __import__("json").loads(
        (tmp_path / "representation_merge_audit.json").read_text()
    )
    assert audit["input_rows"] == 2
    assert audit["unique_rows"] == 2
    assert audit["duplicate_rows"] == 0


def test_instructseg_adapter_preserves_official_expression_and_all_frames(tmp_path):
    image_root = tmp_path / "images"
    (image_root / "v").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (7, 5)).save(image_root / "v" / "000.jpg")
    Image.new("RGB", (7, 5)).save(image_root / "v" / "001.jpg")
    item = {
        "dataset": "long_rvos", "video_id": "v", "object_id": "2",
        "frame_names": ["000", "001"],
        "evaluation_frame_indices": [1], "evaluation_frame_names": ["001"],
        "evaluation_mask_paths": ["/gt/001.png"],
        "expressions": [{"expression_id": "9", "type": "dynamic", "text": "the dog turns"}],
    }
    payload, mapping = prepare_instructseg(
        {"dataset": {"image_root": str(image_root)}, "objects": [item]}, None
    )
    assert payload["videos"][0]["expressions"] == ["the dog turns"]
    assert payload["videos"][0]["file_names"] == ["v/000.jpg", "v/001.jpg"]
    assert mapping[0]["gt_available_to_model"] is False
    assert instructseg_expression_key(item, item["expressions"][0]) == "long_rvos/v/2/9/native"


def test_cross_model_shards_partition_pilot_objects_without_overlap():
    objects = [{"object_id": str(index)} for index in range(70)]
    for selector in (select_instructseg_objects, select_virst_objects):
        shards = [selector(objects, 64, shard_index=index, num_shards=4) for index in range(4)]
        flattened = [item for shard in shards for item in shard]
        assert len(flattened) == 64
        assert {item["object_id"] for item in flattened} == {str(index) for index in range(64)}
        assert all(len(shard) == 16 for shard in shards)


def test_virst_adapter_uses_gt_free_test_schema_and_symlink(tmp_path):
    image_root = tmp_path / "images"
    (image_root / "v").mkdir(parents=True)
    item = {
        "dataset": "long_rvos", "video_id": "v", "object_id": "2",
        "frame_names": ["000", "001"],
        "evaluation_frame_indices": [1], "evaluation_frame_names": ["001"],
        "evaluation_mask_paths": ["/gt/001.png"],
        "vlm_frame_indices": {"16": [0, 1]},
        "expressions": [{"expression_id": "9", "type": "dynamic", "text": "the dog turns"}],
    }
    dataset_root = tmp_path / "adapter"
    mapping = prepare_virst(
        {"dataset": {"image_root": str(image_root)}, "objects": [item]},
        dataset_root,
        None,
    )
    payload = __import__("json").loads(
        (dataset_root / "mevis/valid/meta_expressions.json").read_text()
    )
    expression = payload["videos"]["v"]["expressions"]["object-2__expression-9"]
    assert expression == {"exp": "the dog turns"}
    assert (dataset_root / "mevis/valid/JPEGImages/v").is_symlink()
    assert mapping[0]["gt_available_to_model"] is False

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
    evaluate_entry as evaluate_instructseg_entry,
    expression_key as instructseg_expression_key,
    prepare as prepare_instructseg,
    select_objects as select_instructseg_objects,
)
from projects.evoseg.temporal_compiler.virst_long_rvos_adapter import (
    evaluate_entry as evaluate_virst_entry,
    frame_audit_key,
    load_frame_audit,
    prepare as prepare_virst,
    select_objects as select_virst_objects,
)
from projects.evoseg.temporal_compiler.virst_instrumented_eval import (
    build_frame_audit_record,
    sampling_seed_for_video,
)
from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    audit_loaded_checkpoint,
    candidate_key,
    decode_rle,
    encode_rle,
    load_ground_truth,
    select_pilot_objects,
)
from projects.evoseg.temporal_compiler.extract_qwen_concepts import (
    clean_concept,
    expression_key as qwen_expression_key,
)
from projects.evoseg.temporal_compiler.summarize_cross_model import (
    build_pair_overlap_audit,
    cluster_bootstrap,
    manifest_expressions_by_object,
    manifest_identities,
    read_prediction_files,
    summarize_model,
    select_manifest_objects,
)

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
    shards = [
        select_pilot_objects(values, 32, shard_index=index, num_shards=4)
        for index in range(4)
    ]
    assert all(len(shard) == 8 for shard in shards)
    assert {item["object_id"] for shard in shards for item in shard} == {
        str(index) for index in range(32)
    }


def test_candidate_rle_round_trip():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))


def test_candidate_evaluation_rejects_missing_gt(tmp_path):
    item = {
        "video_id": "v",
        "object_id": "o",
        "evaluation_mask_paths": [str(tmp_path / "missing.png")],
        "evaluation_mask_present": [True],
    }
    try:
        load_ground_truth(item, (5, 7))
    except FileNotFoundError as error:
        assert "evaluation GT availability differs from manifest" in str(error)
    else:
        raise AssertionError("missing GT was silently evaluated as an empty mask")


def test_candidate_evaluation_preserves_manifest_declared_empty_gt(tmp_path):
    item = {
        "video_id": "v",
        "object_id": "o",
        "evaluation_mask_paths": [str(tmp_path / "intentionally_absent.png")],
        "evaluation_mask_present": [False],
    }
    masks = load_ground_truth(item, (5, 7))
    assert masks.shape == (1, 5, 7)
    assert not masks.any()


def test_final_sam31_checkpoint_audit_checks_keys_and_values(tmp_path):
    import torch

    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    audit = audit_loaded_checkpoint(model, checkpoint)
    assert audit["model_key_count"] == 2
    assert audit["missing_keys"] == []
    broken = model.state_dict()
    broken.pop("bias")
    torch.save(broken, checkpoint)
    try:
        audit_loaded_checkpoint(model, checkpoint)
    except RuntimeError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing checkpoint key was not rejected")


def test_qwen_concept_output_cleaning_and_identity():
    item = {"dataset": "d", "video_id": "v", "object_id": "o"}
    expression = {"expression_id": "e"}
    assert qwen_expression_key(item, expression) == "d/v/o/e"
    assert clean_concept('  Concept: "striped tiger."\nExplanation') == "striped tiger"


def test_cluster_bootstrap_preserves_constant_gap():
    rows = [
        {"video_id": "a", "dynamic_minus_static_J_and_F": -0.1},
        {"video_id": "b", "dynamic_minus_static_J_and_F": -0.1},
    ]
    point, low, high = cluster_bootstrap(rows, iterations=20, seed=42)
    assert np.isclose(point, -0.1)
    assert np.isclose(low, -0.1)
    assert np.isclose(high, -0.1)


def test_cross_model_summary_excludes_object_with_missing_official_expression(
    tmp_path,
):
    manifest = {
        "objects": [
            {
                "dataset": "d",
                "video_id": "v",
                "object_id": "1",
                "expressions": [
                    {"expression_id": "s0", "type": "static"},
                    {"expression_id": "s1", "type": "static"},
                    {"expression_id": "d0", "type": "dynamic"},
                ],
            },
            {
                "dataset": "d",
                "video_id": "w",
                "object_id": "2",
                "expressions": [
                    {"expression_id": "s0", "type": "static"},
                    {"expression_id": "d0", "type": "dynamic"},
                ],
            },
        ]
    }
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        ("v", "1", "s0", "static", 0.8),
        # v/1/s1 is deliberately missing: this object must not enter the statistic.
        ("v", "1", "d0", "dynamic", 0.2),
        ("w", "2", "s0", "static", 0.7),
        ("w", "2", "d0", "dynamic", 0.6),
    ]
    predictions.write_text(
        "".join(
            __import__("json").dumps(
                {
                    "key": f"d/{video}/{obj}/{expression}/native",
                    "dataset": "d",
                    "video_id": video,
                    "object_id": obj,
                    "expression_id": expression,
                    "description_type": kind,
                    "status": "success",
                    "J": score,
                    "F": score,
                    "J_and_F": score,
                }
            )
            + "\n"
            for video, obj, expression, kind, score in rows
        )
    )
    summary, pairs, audit = summarize_model(
        {"name": "m", "predictions": str(predictions)},
        manifest_identities(manifest),
        manifest_expressions_by_object(manifest),
        iterations=20,
        seed=42,
    )
    assert summary["eligible_paired_objects"] == 2
    assert summary["paired_objects"] == 1
    assert summary["excluded_incomplete_paired_objects"] == 1
    assert [(row["video_id"], row["object_id"]) for row in pairs] == [("w", "2")]
    assert audit["incomplete_paired_objects"][0][
        "missing_static_expression_ids"
    ] == ["s1"]


def test_cross_model_pilot_selection_and_pair_overlap_are_explicit():
    manifest = {
        "objects": [
            {
                "dataset": "d",
                "video_id": f"v{index}",
                "object_id": str(index),
                "expressions": [
                    {"expression_id": "s", "type": "static"},
                    {"expression_id": "d", "type": "dynamic"},
                ],
            }
            for index in range(3)
        ]
    }
    selected = select_manifest_objects(manifest, 2)
    expected = manifest_expressions_by_object(selected)
    pairs = [
        {"model": "a", "dataset": "d", "video_id": "v0", "object_id": "0"},
        {"model": "a", "dataset": "d", "video_id": "v1", "object_id": "1"},
        {"model": "b", "dataset": "d", "video_id": "v0", "object_id": "0"},
    ]
    audit = build_pair_overlap_audit(pairs, ["a", "b"], expected)
    assert len(selected["objects"]) == 2
    assert audit["complete_paired_objects_by_model"] == {"a": 2, "b": 1}
    assert audit["shared_complete_paired_objects"] == 1
    assert audit["all_models_have_identical_complete_object_set"] is False


def test_cross_model_reads_multiple_prediction_shards_without_silent_duplicates(
    tmp_path,
):
    shard_zero = tmp_path / "zero.jsonl"
    shard_one = tmp_path / "one.jsonl"
    shard_zero.write_text('{"key":"a"}\n')
    shard_one.write_text('{"key":"b"}\n')
    records, paths = read_prediction_files([str(shard_zero), str(shard_one)])
    assert [record["key"] for record in records] == ["a", "b"]
    assert paths == [str(shard_zero), str(shard_one)]
    shard_one.write_text('{"key":"a"}\n')
    try:
        read_prediction_files([str(shard_zero), str(shard_one)])
    except RuntimeError as error:
        assert "duplicate prediction key across files" in str(error)
    else:
        raise AssertionError("cross-shard duplicate prediction key was accepted")


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
        "evaluation_mask_present": [True],
        "expressions": [
            {"expression_id": "9", "type": "dynamic", "text": "the dog turns"},
            {"expression_id": "10", "type": "static", "text": "the brown dog"},
        ],
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


def test_virst_frame_audit_records_realized_vlm_and_sam_indices(tmp_path):
    class TensorStub:
        def __init__(self, shape, nonzero=0):
            self.shape = shape
            self.nonzero = nonzero

        def count_nonzero(self):
            return ScalarStub(self.nonzero)

    class ScalarStub:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    record = build_frame_audit_record(
        7,
        {
            "frame_ids": [[0, 4, 9]],
            "images_clip": TensorStub((4, 3, 448, 448)),
            "images_sam": TensorStub((3, 3, 1024, 1024)),
            "masks": [TensorStub((1, 3, 1024, 1024))],
            "video_paths": ["/data/v"],
            "exp_ids": ["e"],
            "image_path": (
                "/data/v/000.jpg,/data/v/002.jpg,/data/v/004.jpg,/data/v/009.jpg"
            ),
            "_evoseg_vlm_original_frame_indices": [0, 2, 4, 9],
            "_evoseg_sampling_seed": sampling_seed_for_video("v"),
        },
    )
    assert record["model_input_frame_indices"] == [0, 2, 4, 9]
    assert record["vlm_frame_indices"] == [0, 2, 4, 9]
    assert record["sam_frame_indices"] == [0, 4, 9]
    assert record["vlm_frame_count"] == 4
    assert record["sam_frame_count"] == 3
    assert record["same_video_expression_sampling_locked"] is True
    assert record["sampling_seed"] == sampling_seed_for_video("v")
    assert record["gt_available_to_model"] is False
    assert record["video_id"] == "v"
    assert record["output_expression_ids"] == ["e"]
    audit_path = tmp_path / "frame_audit.jsonl"
    payload = __import__("json").dumps(record) + "\n"
    audit_path.write_text(payload + payload)
    loaded = load_frame_audit(audit_path)
    assert loaded[frame_audit_key("v", "e")]["model_input_frame_indices"] == [
        0, 2, 4, 9
    ]
    inconsistent = {
        **record,
        "model_input_frame_indices": [1, 2, 4, 9],
        "vlm_frame_indices": [1, 2, 4, 9],
    }
    audit_path.write_text(payload + __import__("json").dumps(inconsistent) + "\n")
    retried = load_frame_audit(audit_path)[frame_audit_key("v", "e")]
    assert retried["model_input_frame_indices"] == [1, 2, 4, 9]
    assert retried["superseded_frame_audit_attempts"] == 1


def test_virst_sampling_seed_is_stable_by_video():
    assert sampling_seed_for_video("video-a") == sampling_seed_for_video("video-a")
    assert sampling_seed_for_video("video-a") != sampling_seed_for_video("video-b")


def test_virst_adapter_uses_gt_free_test_schema_and_symlink(tmp_path):
    image_root = tmp_path / "images"
    (image_root / "v").mkdir(parents=True)
    item = {
        "dataset": "long_rvos", "video_id": "v", "object_id": "2",
        "frame_names": ["000", "001"],
        "evaluation_frame_indices": [1], "evaluation_frame_names": ["001"],
        "evaluation_mask_paths": ["/gt/001.png"],
        "evaluation_mask_present": [True],
        "vlm_frame_indices": {"16": [0, 1]},
        "expressions": [{"expression_id": "9", "type": "dynamic", "text": "the dog turns"}],
    }
    dataset_root = tmp_path / "adapter"
    mapping = prepare_virst(
        {"dataset": {"image_root": str(image_root)}, "objects": [item]},
        dataset_root,
        None,
        max_expressions_per_object=1,
    )
    payload = __import__("json").loads(
        (dataset_root / "mevis/valid/meta_expressions.json").read_text()
    )
    expression = payload["videos"]["v"]["expressions"]["object-2__expression-9"]
    assert expression == {"exp": "the dog turns"}
    assert list(payload["videos"]["v"]["expressions"]) == [
        "object-2__expression-9"
    ]
    assert len(mapping) == 1
    assert (dataset_root / "mevis/valid/JPEGImages/v").is_symlink()
    assert mapping[0]["gt_available_to_model"] is False


def test_cross_model_adapters_honor_manifest_declared_empty_gt(tmp_path):
    from PIL import Image

    entry = {
        "key": "long_rvos/v/2/9/native",
        "dataset": "long_rvos",
        "video_id": "v",
        "object_id": "2",
        "expression_id": "9",
        "description_type": "dynamic",
        "expression": "the dog turns",
        "output_video": "v__2__9",
        "output_expression": "0",
        "model_input_frame_indices": [0],
        "evaluation_frame_indices": [0],
        "evaluation_frame_names": ["000"],
        "evaluation_mask_paths": [str(tmp_path / "intentionally_absent.png")],
        "evaluation_mask_present": [False],
        "native_reference_frame_num": 4,
        "manifest_vlm_frame_indices_n16": [0],
    }
    prediction_root = tmp_path / "predictions"
    prediction_dir = prediction_root / "v__2__9" / "0"
    prediction_dir.mkdir(parents=True)
    Image.new("L", (7, 5), 0).save(prediction_dir / "000.png")
    assert evaluate_instructseg_entry(entry, prediction_root)["status"] == "success"
    assert evaluate_virst_entry(entry, prediction_root)["status"] == "success"

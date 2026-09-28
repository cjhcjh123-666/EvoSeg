import numpy as np

from projects.evoseg.temporal_grounding_interface.run_dynamic_interface import (
    tracking_signature,
)
from projects.evoseg.temporal_grounding_mechanism.build_train_manifest import (
    uniform_indices,
)
from projects.evoseg.temporal_grounding_mechanism.common import (
    object_weighted,
    source_video_bootstrap,
)
from projects.evoseg.temporal_grounding_mechanism.point_executor import (
    rebuild_tracking_cache,
)
from projects.evoseg.temporal_grounding_mechanism.pixel_execution import (
    plan_signature,
    tight_box,
)
from projects.evoseg.temporal_grounding_mechanism.train_probe import ProbeFactory


def test_train_frame_sampling_is_uniform_and_bounded():
    assert uniform_indices(4, 64) == [0, 1, 2, 3]
    values = uniform_indices(235, 64)
    assert len(values) == len(set(values)) == 64
    assert values[0] == 0 and values[-1] == 234


def test_statistics_weight_objects_before_source_video_bootstrap():
    rows = [
        {"video_id": "a", "object_id": "1", "description_type": "dynamic", "left": 1, "right": 0},
        {"video_id": "a", "object_id": "1", "description_type": "dynamic", "left": 0, "right": 0},
        {"video_id": "b", "object_id": "2", "description_type": "dynamic", "left": 0, "right": 0},
    ]
    objects = object_weighted(rows, ["left", "right"])
    assert len(objects) == 2
    assert np.isclose(objects[0]["left"], 0.5)
    result = source_video_bootstrap(rows, "left", "right", iterations=20)
    assert result["objects"] == result["videos"] == 2
    assert np.isclose(result["mean"], 0.25)


def test_probe_architecture_is_shared_and_accepts_candidate_sequences():
    import torch

    static = ProbeFactory.build(query_dim=4, visual_dim=6, hidden=8)
    temporal = ProbeFactory.build(query_dim=4, visual_dim=6, hidden=8)
    assert sum(x.numel() for x in static.parameters()) == sum(
        x.numel() for x in temporal.parameters()
    )
    scores = static(torch.randn(3, 4), torch.randn(3, 4, 4), torch.randn(3, 4, 6))
    assert scores.shape == (3,)


def test_tracking_cache_signature_uses_only_public_pixel_prompts():
    base = {
        "anchor_frame_index": 10,
        "positive_point_relative_xy": [0.25, 0.75],
        "update_applied": True,
        "state_kind": "static",
        "candidate_scores": [0.1, 0.9],
    }
    changed_metadata = {**base, "state_kind": "temporal", "candidate_scores": [0.8, 0.2]}
    assert tracking_signature("v", [base]) == tracking_signature("v", [changed_metadata])
    changed_point = {**base, "positive_point_relative_xy": [0.2, 0.7]}
    assert tracking_signature("v", [base]) != tracking_signature("v", [changed_point])


def test_point_cache_is_rebuilt_from_successful_resume_records():
    plan = {
        "identity": "long_rvos/v/1/0",
        "condition": "ORACLE_ID_POINT",
        "video_id": "v",
        "object_id": "1",
        "plan": [
            {
                "anchor_frame_index": 10,
                "positive_point_relative_xy": [0.25, 0.75],
                "update_applied": True,
            }
        ],
    }
    previous = {
        "identity": plan["identity"],
        "condition": plan["condition"],
        "status": "success",
        "J": 0.5,
        "F": 0.6,
        "J_and_F": 0.55,
        "prompt_updates": 1,
        "latency_seconds_synchronized": 1.0,
        "peak_memory_bytes": 2,
        "session_api_compat": "official",
    }
    cache = rebuild_tracking_cache(
        [plan], {(plan["identity"], plan["condition"]): previous}
    )
    signature = "1:" + tracking_signature("v", plan["plan"])
    assert cache[signature]["J_and_F"] == 0.55
    assert cache[signature]["reused_source_condition"] == "ORACLE_ID_POINT"


def test_pixel_prompt_geometry_is_candidate_derived_and_normalized():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:6, 5:15] = True
    assert tight_box(mask) == [0.25, 0.2, 0.5, 0.4]
    assert tight_box(np.zeros_like(mask)) is None


def test_pixel_execution_signature_fixes_candidate_identity_and_prompt_form():
    row = {
        "video_id": "v",
        "object_id": "1",
        "plan": [{
            "anchor_frame_index": 10,
            "selected_candidate_object_id": 7,
            "positive_point_relative_xy": [0.25, 0.75],
        }],
    }
    mask = np.ones((4, 4), dtype=bool)
    assert plan_signature(row, "point") != plan_signature(row, "box", [mask])
    changed = {**row, "plan": [{**row["plan"][0], "selected_candidate_object_id": 8}]}
    assert plan_signature(row, "box", [mask]) != plan_signature(changed, "box", [mask])

from projects.evoseg.temporal_compiler.extract_sa2va_representations import (
    select_objects,
    stable_key,
)
from projects.evoseg.temporal_compiler.analyze_sa2va_representations import distances
from projects.evoseg.temporal_compiler.sam31_candidate_protocol import (
    candidate_key,
    decode_rle,
    encode_rle,
    select_pilot_objects,
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


def test_candidate_rle_round_trip():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))

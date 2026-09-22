from projects.evoseg.temporal_compiler.extract_sa2va_representations import (
    select_objects,
    stable_key,
)
from projects.evoseg.temporal_compiler.analyze_sa2va_representations import distances

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

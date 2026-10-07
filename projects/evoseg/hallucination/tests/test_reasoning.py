import numpy as np

from projects.evoseg.hallucination.prepare_reasoning import annotation_masks
from projects.evoseg.hallucination.eval_fresh import metric_values


def test_original_ignore_polygon_excluded_from_target_and_score():
    annotation = {'shapes': [
        {'label': 'target', 'points': [[0, 0], [9, 0], [9, 9], [0, 9]]},
        {'label': 'ignore', 'points': [[3, 3], [5, 3], [5, 5], [3, 5]]},
        {'label': 'flag', 'points': [[0, 0], [9, 0], [9, 9], [0, 9]]},
    ]}
    target, valid = annotation_masks(annotation, (10, 10))
    assert target.sum() == 91 and valid.sum() == 91
    prediction = np.ones((10, 10), dtype=bool)
    assert metric_values(target, prediction, False, valid)['iou'] == 1.
    assert metric_values(target, prediction, False)['iou'] == .91

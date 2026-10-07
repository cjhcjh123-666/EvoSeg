from types import SimpleNamespace

import pytest

from projects.evoseg.hallucination.continue_training import preservation_guard
from projects.evoseg.hallucination import eval_fresh
from projects.evoseg.hallucination.run_extended import matched_candidate


def cases(drop=0.):
    baseline, candidate, images = {}, {}, {}
    for i in range(40):
        empty = i >= 30
        baseline[str(i)] = {'empty_gt': empty, 'iou': 1. if empty else .8,
                            'predicted_empty': empty}
        candidate[str(i)] = {**baseline[str(i)], 'iou': 1. if empty else .8 - drop}
        images[str(i)] = i // 2
    return baseline, candidate, images


def test_guard_passes_unchanged_and_detects_clear_positive_regression():
    assert preservation_guard(*cases())['qualified']
    value = preservation_guard(*cases(.02))
    assert not value['qualified']
    assert value['positive_iou_delta'] == pytest.approx(-.02)
    assert value['positive_image_clusters'] == 15
    assert not value['selection_uses_public_benchmark']


def test_guard_requires_exact_paired_coverage():
    a, b, ids = cases()
    b.pop('0')
    with pytest.raises(ValueError, match='coverage'):
        preservation_guard(a, b, ids)


def test_expanded_holdout_does_not_trim_unequal_buckets(monkeypatch):
    rows = [{'id': str(i), 'holdout': True, 'bucket': 'positive' if i < 7 else 'empty'}
            for i in range(10)]
    monkeypatch.setattr(eval_fresh, 'cached_records', lambda cache: rows)
    args = SimpleNamespace(benchmark='holdout', limit=None, cache='unused')
    assert len(eval_fresh.evaluation_records(args)) == 10


def test_final_comparison_uses_the_same_training_step_for_both_arms():
    history = []
    for step, score, safe in [(100, .8, True), (300, .9, False), (600, .85, True)]:
        history.append({'requested_step': step,
            'scores': {'interaction': {'gIoU': score}, 'full_view': {'gIoU': .7}},
            'guards': {'interaction': {'qualified': safe}, 'full_view': {'qualified': True}},
            'checkpoints': {'interaction': f'method_{step}', 'full_view': f'control_{step}'}})
    candidate = matched_candidate({'history': history})
    assert candidate['selection_step'] == 600
    assert candidate['checkpoints'] == {'interaction': 'method_600', 'full_view': 'control_600'}
    assert not candidate['selection_uses_public_benchmark']

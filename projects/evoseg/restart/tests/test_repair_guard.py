import copy

from projects.evoseg.restart.native_data import split_video_ids
from projects.evoseg.restart.repair_pilot import heldout_keys, passes_guard


def score(overall=.7, positive=.75, missing=.04):
    return {'metrics': {'j_and_f': overall, 'target_present': {'j_and_f': positive}},
            'presence': {'empty_on_present_fraction': missing}}


def test_guard_rejects_positive_collapse_even_when_empty_queries_improve_overall():
    baseline = score()
    assert passes_guard(baseline, score())
    assert not passes_guard(baseline, score(overall=.8, positive=.70))
    assert not passes_guard(baseline, score(missing=.08))
    assert not passes_guard(baseline, score(overall=.68))
    incomplete = score()
    incomplete['presence']['empty_on_present_fraction'] = None
    assert not passes_guard(baseline, incomplete)


def test_expanded_diagnostic_uses_only_heldout_training_videos():
    videos = {f'v{i}': {'expressions': {'0': {'anno_id': [1]}, '1': {'anno_id': [2]},
                                       '2': {'anno_id': []}}} for i in range(100)}
    original = copy.deepcopy(videos)
    train, held = split_video_ids(videos)
    keys = heldout_keys(videos)
    assert len(keys) == 3 * len(held)
    assert {key[0] for key in keys} == set(held)
    assert {key[0] for key in keys}.isdisjoint(train)
    assert len(keys) == len({tuple(key) for key in keys})
    assert heldout_keys(videos) == keys and videos == original

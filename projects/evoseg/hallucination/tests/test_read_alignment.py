import inspect
from types import SimpleNamespace

import pytest
import torch

from projects.evoseg.hallucination.read_alignment import (
    SasPConfig, aligned_predictions, merged_image_features,
    native_prediction_states, similarity_as_points,
)
from projects.evoseg.hallucination.run_read_alignment import aggregate, select_records


def test_rectangular_grid_points_are_xy_and_inside_direct_resize():
    features = torch.arange(8.).reshape(8, 1)
    points, audit = similarity_as_points(features, torch.ones(1), (2, 4), (16, 32),
        config=SasPConfig(downsample=1, max_points=1), sam_size=64)
    assert points['point_labels'].tolist() == [[1]]
    x, y = points['point_coords'][0, 0].tolist()
    # Largest activation is lower right, not a transposed raster coordinate.
    assert 48 < x < 64 and 32 < y < 64
    assert audit['grid'] == [2, 4] and not audit['uses_ground_truth']


def test_flat_map_is_native_noop_not_nan_or_invented_foreground():
    points, audit = similarity_as_points(torch.ones(4, 3), torch.ones(3), (2, 2), (8, 8))
    assert points is None and audit['skip_reason'] == 'flat_similarity'


def test_points_have_feature_gradients_and_published_budget_order():
    torch.manual_seed(19)
    features = torch.randn(16, 4, requires_grad=True)
    semantic = torch.randn(4, requires_grad=True)
    points, audit = similarity_as_points(features, semantic, (4, 4), (12, 20),
        config=SasPConfig(downsample=1, max_points=10))
    labels = points['point_labels'][0].tolist()
    assert len(labels) <= 10 and labels == sorted(labels, reverse=True)
    points['point_coords'].square().sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert torch.isfinite(semantic.grad).all() and semantic.grad.abs().sum() > 0
    assert sum(audit[k] for k in ('positive', 'negative', 'neutral')) == len(labels)


def test_similarity_validation_refuses_unknown_geometry_and_nonfinite():
    with pytest.raises(ValueError, match='grid'):
        similarity_as_points(torch.ones(3, 4), torch.ones(4), (2, 2), (4, 4))
    with pytest.raises(ValueError, match='non-finite'):
        similarity_as_points(torch.full((4, 3), float('nan')), torch.ones(3), (2, 2), (4, 4))
    with pytest.raises(ValueError, match='threshold'):
        SasPConfig(negative_threshold=.9)


def test_merge_geometry_requires_verified_image_token_count():
    config = SimpleNamespace(image_token_id=10, vision_config=SimpleNamespace(spatial_merge_size=2))
    inputs = SimpleNamespace(input_ids=torch.tensor([[0, 10, 10, 10, 10, 10, 10, 2]]),
        image_grid_thw=torch.tensor([[1, 4, 6]]))
    hidden = torch.randn(1, 8, 4)
    features, grid = merged_image_features(hidden, inputs, config)
    assert grid == (2, 3)
    torch.testing.assert_close(features, hidden[0, 1:7])
    inputs.image_grid_thw = torch.tensor([[1, 4, 8]])
    with pytest.raises(ValueError, match='token count'):
        merged_image_features(hidden, inputs, config)


def test_native_hidden_alignment_uses_predictor_state_not_following_token():
    pieces = ['hello', '<|mt_start|>', '<|mt_0001|>', '<|mt_0258|>', '<|mt_end|>', 'done']
    processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: pieces))
    generated = SimpleNamespace(sequences=torch.arange(8)[None],
        hidden_states=[(torch.full((1, 1, 4), float(i)),) for i in range(6)])
    states = native_prediction_states(generated, 2, processor, [[1, 2]])
    torch.testing.assert_close(states[0], torch.full((4,), 4.))
    assert native_prediction_states(generated, 2, processor, [[3, 2]]) == [None]


def test_image_query_only_inference_contract():
    assert list(inspect.signature(aligned_predictions).parameters) == ['runtime', 'image_path', 'query']
    assert list(inspect.signature(similarity_as_points).parameters) == [
        'image_features', 'semantic', 'grid', 'image_shape', 'config', 'sam_size']


def test_sam_native_path_and_multimask_policy_do_not_change_with_points():
    from projects.samtok.models.sam2 import SAM2Model
    captured = []

    def forward(**kwargs):
        captured.append(kwargs)
        return (None, None, None, torch.ones(1, 1, 2, 2), None, None, None)

    sam = SimpleNamespace(directly_add_no_mem_embed=True, no_mem_embed=torch.zeros(1, 1, 2),
        _forward_sam_heads=forward, _use_multimask=lambda **kw: kw['point_inputs'] is None)
    wrapper = SimpleNamespace(sam2_model=sam, hidden_dim=2)
    states = {'current_vision_feats': [torch.zeros(4, 1, 2)], 'feat_sizes': [(2, 2)]}
    language = torch.zeros(1, 1, 2)
    a = SAM2Model.inject_language_embd(wrapper, states, language, nf_nobj=(1, 1))
    points = {'point_coords': torch.ones(1, 2, 2), 'point_labels': torch.tensor([[1, 0]])}
    b = SAM2Model.inject_language_embd(wrapper, states, language, nf_nobj=(1, 1), point_inputs=points)
    torch.testing.assert_close(a, b)
    assert captured[0]['point_inputs'] is None and captured[1]['point_inputs'] is points
    assert captured[0]['multimask_output'] == captured[1]['multimask_output']


def test_private_record_selection_is_balanced_original_and_score_independent():
    rows = [{'id': str(i), 'holdout': True, 'source_split': 'train',
             'generated_query': False, 'pseudo_label': False, 'bucket': 'a' if i < 4 else 'b'}
            for i in range(6)]
    assert [r['id'] for r in select_records(rows, 4)] == ['0', '4', '1', '5']
    rows[0]['source_split'] = 'val'
    with pytest.raises(ValueError, match='TRAIN'):
        select_records(rows, 4)


def test_report_includes_empty_outputs_not_just_candidate_bearing_cases():
    metric = {'iou': 1., 'intersection': 0, 'union': 0, 'predicted_empty': True}
    case = {'bucket': 'gref_empty', 'inference': {'native_empty': True, 'points': []},
            'metrics': {'native': metric, 'read_style': metric}}
    report = aggregate([case])
    assert report['cases'] == 1 and report['native_empty_cases'] == 1
    assert report['per_bucket']['gref_empty']['native']['mean_iou'] == 1.
    assert not report['public_benchmark'] and not report['sota_claimed']

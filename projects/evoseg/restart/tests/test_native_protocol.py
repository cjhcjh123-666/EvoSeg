import pytest

from projects.sasasa2va.evaluation.protocol import resolve_dataset_info


DEFAULT = {'data_root': 'old', 'image_folder': 'old/JPEGImages',
           'expression_file': 'old/meta_expressions.json', 'mask_file': 'old/mask_dict.json',
           'selected_frame_file': 'missing_q_frames.json'}


def test_uniform_does_not_read_q_frame_indices():
    assert resolve_dataset_info(DEFAULT, mode='uniform')['selected_frame_file'] is None
    assert DEFAULT['selected_frame_file'] == 'missing_q_frames.json'


def test_explicit_public_split_root(tmp_path):
    result = resolve_dataset_info(DEFAULT, mode='uniform', data_root=tmp_path)
    assert result['image_folder'] == str(tmp_path / 'JPEGImages')
    assert result['expression_file'] == str(tmp_path / 'meta_expressions.json')
    assert result['mask_file'] == str(tmp_path / 'mask_dict.json')


def test_blind_test_split_still_has_no_masks(tmp_path):
    result = resolve_dataset_info(dict(DEFAULT, mask_file=None), mode='uniform', data_root=tmp_path)
    assert result['mask_file'] is None


def test_explicit_dataset_version_metadata(tmp_path):
    expressions = tmp_path / 'meta_expressions_v2.json'
    result = resolve_dataset_info(DEFAULT, mode='uniform', data_root=tmp_path,
                                  expression_file=expressions)
    assert result['expression_file'] == str(expressions)


def test_q_frame_requires_existing_indices(tmp_path):
    with pytest.raises(ValueError, match='existing'):
        resolve_dataset_info(DEFAULT, mode='q_frame')
    indices = tmp_path / 'indices.json'
    indices.write_text('{}')
    result = resolve_dataset_info(DEFAULT, mode='q_frame', selected_frame_file=indices)
    assert result['selected_frame_file'] == str(indices)


def test_no_accidental_q_frame_or_invalid_mode():
    with pytest.raises(ValueError, match='only'):
        resolve_dataset_info(DEFAULT, mode='uniform', selected_frame_file='anything.json')
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_dataset_info(DEFAULT, mode='default')

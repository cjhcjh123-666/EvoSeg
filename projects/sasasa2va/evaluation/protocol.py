"""Resolve native evaluation inputs without changing model preprocessing."""
from pathlib import Path


INFERENCE_MODES = ('uniform', 'uniform_plus', 'q_frame', 'wrap_around', 'wrap_around_plus')


def resolve_dataset_info(info, *, mode, data_root=None, expression_file=None, selected_frame_file=None):
    if mode not in INFERENCE_MODES:
        raise ValueError(f'Unsupported native inference mode: {mode}')
    resolved = dict(info)
    if data_root is not None:
        root = Path(data_root)
        resolved.update(data_root=str(root), image_folder=str(root / 'JPEGImages'),
                        expression_file=str(root / 'meta_expressions.json'))
        if info.get('mask_file') is not None:
            resolved['mask_file'] = str(root / 'mask_dict.json')
    if expression_file is not None:
        resolved['expression_file'] = str(expression_file)
    if mode == 'q_frame':
        selected = selected_frame_file or info.get('selected_frame_file')
        if not selected or not Path(selected).is_file():
            raise ValueError('q_frame requires an existing selected-frame JSON file')
        resolved['selected_frame_file'] = str(selected)
    else:
        if selected_frame_file is not None:
            raise ValueError('Selected-frame files are used only in q_frame mode')
        resolved['selected_frame_file'] = None
    return resolved

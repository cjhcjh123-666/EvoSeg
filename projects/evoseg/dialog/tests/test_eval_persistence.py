import json

from projects.evoseg.dialog.eval_multiturn import save_case


def test_first_case_creates_nested_result_directory_atomically(tmp_path):
    destination = tmp_path / 'refcoco/cases/000000.json'
    values = {'history_mode': 'gt_history', 'rounds': [{'round': 1, 'count': 1}]}
    save_case(destination, values)
    assert json.loads(destination.read_text()) == values
    assert not destination.with_suffix('.tmp').exists()

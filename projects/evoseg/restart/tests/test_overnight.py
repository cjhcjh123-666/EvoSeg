from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from projects.evoseg.restart.native_eval import expression_tasks, merge_predictions, native_query
from projects.evoseg.restart.overnight import next_morning_deadline


def test_next_nine_am_is_exact_beijing_deadline():
    timezone = ZoneInfo('Asia/Shanghai')
    now = datetime(2026, 10, 6, 0, 48, tzinfo=timezone)
    assert datetime.fromtimestamp(next_morning_deadline(now), timezone) == datetime(2026, 10, 6, 9, tzinfo=timezone)
    later = datetime(2026, 10, 6, 10, tzinfo=timezone)
    assert datetime.fromtimestamp(next_morning_deadline(later), timezone).day == 7


def test_native_v2_questions_keep_text_and_receive_images():
    assert native_query('what causes that rabbit to jump?') == '<image>\nwhat causes that rabbit to jump?'
    assert native_query('the running dog') == '<image>\nPlease segment the running dog.'


def test_eval_shards_cover_once_without_padding():
    metadata = {'v': {'expressions': {str(i): {} for i in range(11)}}}
    tasks = expression_tasks(metadata)
    shards = [tasks[rank::8] for rank in range(8)]
    combined = [key for shard in shards for key in shard]
    assert len(combined) == len(set(combined)) == 11


def test_merge_rejects_short_frame_coverage(tmp_path):
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps({'videos': {'v': {'frames': ['0', '1'], 'expressions': {'0': {}}}}}))
    output = tmp_path / 'output'
    (output / 'cases/v').mkdir(parents=True)
    (output / 'cases/v/0.json').write_text(json.dumps({'frames': ['0'], 'prediction_masks': [{}]}))
    with pytest.raises(RuntimeError, match='frame inventory'):
        merge_predictions(meta, output)

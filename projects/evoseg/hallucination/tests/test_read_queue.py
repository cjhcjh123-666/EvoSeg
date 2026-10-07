from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from projects.evoseg.hallucination.queue_read_alignment import execute, stop_owned


def args(tmp_path):
    return SimpleNamespace(output=str(tmp_path / 'queue'), after_report=str(tmp_path / 'smoke/REPORT.json'),
        deadline=(datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
        limit=96, cache=None, python='not-launched')


def test_failed_smoke_does_not_launch_larger_run(tmp_path, monkeypatch):
    (tmp_path / 'smoke').mkdir()
    (tmp_path / 'smoke/ERROR.json').write_text('{}')
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='smoke failed'):
        execute(args(tmp_path))
    status = json.loads((tmp_path / 'queue/QUEUE_STATUS.json').read_text())
    assert status['status'] == 'FAILED_PREDICTIONS_PRESERVED'


def test_smoke_without_real_points_does_not_launch(tmp_path, monkeypatch):
    (tmp_path / 'smoke').mkdir()
    (tmp_path / 'smoke/REPORT.json').write_text(json.dumps({'candidates_with_points': 0}))
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='point prompts'):
        execute(args(tmp_path))


def test_deadline_is_checked_even_if_smoke_already_exists(tmp_path, monkeypatch):
    (tmp_path / 'smoke').mkdir()
    (tmp_path / 'smoke/REPORT.json').write_text(json.dumps({'candidates_with_points': 1}))
    value = args(tmp_path)
    value.deadline = '2020-01-01T00:00:00+00:00'
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('must not launch'))
    execute(value)
    status = json.loads((tmp_path / 'queue/QUEUE_STATUS.json').read_text())
    assert status['status'] == 'DEADLINE_REACHED_PARTIAL_RESULTS'


def test_finished_own_child_is_not_signalled(monkeypatch):
    monkeypatch.setattr('os.killpg', lambda *a: pytest.fail('must not signal'))
    stop_owned(SimpleNamespace(poll=lambda: 0))

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
from types import SimpleNamespace

from projects.evoseg.hallucination import share_read_gpu as share


def test_process_ownership_requires_exact_module_and_output(monkeypatch):
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'python\0-m\0the.module\0--output\0/our/output-extra\0')
    assert not share.owns_command(123, 'the.module', Path('/our/output'))
    assert share.owns_command(123, 'the.module', Path('/our/output-extra'))
    assert not share.owns_command(123, 'another.module', Path('/our/output-extra'))


def test_public_supervisor_is_restored_if_shared_worker_launch_fails(tmp_path, monkeypatch):
    public, cpu = tmp_path / 'public', tmp_path / 'cpu'
    public.mkdir(); cpu.mkdir()
    (public / 'RECOVERY_STATUS.json').write_text(json.dumps({'prediction_directory': '/eval', 'devices': [0]}))
    (cpu / 'QUEUE_STATUS.json').write_text(json.dumps({'child_pid': 222}))
    args = SimpleNamespace(output=str(tmp_path / 'share'), public_output=str(public), cpu_output=str(cpu),
        public_supervisor=100, cpu_supervisor=200, gpu=0, limit=96, python='python', max_seconds=900,
        deadline=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    calls = []
    monkeypatch.setattr(share, 'owns_command', lambda *a: True)
    monkeypatch.setattr(share, 'gpu_inventory', lambda: {0: ('GPU-a', 65000)})
    monkeypatch.setattr(share, 'gpu_processes', lambda: [('GPU-a', 111, 19000)])
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    monkeypatch.setattr(share.os, 'kill', lambda pid, sig: calls.append((pid, sig)))

    def fail(*a, **k):
        raise RuntimeError('deliberate launch failure')

    monkeypatch.setattr(share.subprocess, 'Popen', fail)
    share.execute(args)
    assert calls == [(200, signal.SIGTERM), (222, signal.SIGTERM),
                     (100, signal.SIGSTOP), (100, signal.SIGCONT)]
    state = json.loads((tmp_path / 'share/SHARE_STATUS.json').read_text())
    assert state['public_supervisor_restored'] and not state['public_workers_paused']
    assert state['status'] == 'PARTIAL_PREDICTIONS_PRESERVED'

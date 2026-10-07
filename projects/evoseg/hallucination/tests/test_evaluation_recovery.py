from types import SimpleNamespace
from pathlib import Path

from projects.evoseg.hallucination import recover_evaluations as recovery


def test_falls_back_to_free_cards_without_using_busy_cards(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='0, 81220\n1, 81000\n2, 23000\n3, 79000\n4, 81000\n'))
    assert recovery.free_devices() == [0, 1, 3, 4]


def test_foreign_task_on_unselected_gpu_does_not_stop_our_job(monkeypatch):
    def command(args, **kwargs):
        if '--query-gpu=index,uuid' in args:
            return SimpleNamespace(stdout='0, GPU-a\n1, GPU-b\n')
        return SimpleNamespace(stdout='GPU-a, 123, 22000\nGPU-b, 456, 57000\n')
    monkeypatch.setattr(recovery.subprocess, 'run', command)
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'python eval --output /test/run')
    assert recovery.foreign_gpu_processes(Path('/test/run'), [0]) == []


def test_foreign_task_on_selected_gpu_is_detected(monkeypatch):
    def command(args, **kwargs):
        if '--query-gpu=index,uuid' in args:
            return SimpleNamespace(stdout='0, GPU-a\n')
        return SimpleNamespace(stdout='GPU-a, 456, 57000\n')
    monkeypatch.setattr(recovery.subprocess, 'run', command)
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'python unrelated-job')
    assert recovery.foreign_gpu_processes(Path('/test/run'), [0]) == [456]

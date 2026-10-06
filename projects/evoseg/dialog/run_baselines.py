"""Detached public baseline chain; never conflates GT history with rollout."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from projects.evoseg.restart.overnight import available_gpu_indices
from projects.evoseg.restart.prepare_foundation import atomic_json


def execute(args):
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'BASELINES.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / 'BASELINE_STATUS.json'
    state = {'status': 'PREPARING', 'pid': os.getpid(), 'training_started': False,
             'started_at_unix': time.time(), 'source_report_comparability': 'pending', 'success': False}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(status_path, state)
        print(json.dumps(state), flush=True)

    def wait_gpus():
        while True:
            indices = available_gpu_indices(args.gpus)
            if indices:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, indices))
                update(gpu_indices=indices)
                return
            update(status='WAITING_FOR_FREE_GPUS')
            time.sleep(30)

    def child(name, command):
        update(status=name, stage_log=str(output / (name + '.log')))
        with (output / (name + '.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=process.pid)
            while process.poll() is None:
                time.sleep(15)
                update(child_pid=process.pid)
            if process.returncode:
                raise RuntimeError(f'{name} failed with exit {process.returncode}; inspect stage log')
        update(child_pid=None)

    def evaluate(mode, *, limit=None, distributed=True):
        name = mode if limit is None else mode + '_diagnostic'
        destination = output / name
        arguments = ['--history-mode', mode, '--output', str(destination)]
        if limit is not None:
            arguments += ['--limit', str(limit)]
        if distributed:
            command = [args.python, '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc_per_node={args.gpus}', '-m', 'projects.evoseg.dialog.eval_multiturn', *arguments]
        else:
            command = [args.python, '-m', 'projects.evoseg.dialog.eval_multiturn', *arguments]
        wait_gpus()
        child(name, command)
        child(name + '_summary', [args.python, '-m', 'projects.evoseg.dialog.eval_multiturn',
                                  *arguments, '--summarize'])
        metrics = json.loads((destination / 'METRICS.json').read_text())
        return destination, metrics

    try:
        update()
        # Each protocol must first finish full multi-round dialogues, not only
        # the first-round image demo. Diagnostics are never called benchmarks.
        for mode in ('gt_history', 'predicted_history'):
            _, score = evaluate(mode, limit=2, distributed=False)
            rounds = [item for dataset in score['datasets'].values() for item in dataset['rounds'].values()]
            if sum(item['count'] for item in rounds) == 0:
                raise RuntimeError('empty multi-turn diagnostic')
            if sum(item['invalid_outputs'] for item in rounds) > .1 * sum(item['count'] for item in rounds):
                raise RuntimeError('too many invalid mask token outputs in native multi-turn diagnostic')
        completed = {}
        for mode in ('gt_history', 'predicted_history'):
            destination, score = evaluate(mode)
            total = sum(sum(value['count'] for value in dataset['rounds'].values())
                        for dataset in score['datasets'].values())
            if total != 17349:
                raise RuntimeError('full public baseline must cover all 17,349 validation rounds')
            if mode == 'predicted_history':
                for path in destination.glob('*/cases/*.json'):
                    if json.loads(path.read_text())['gt_mask_encodings_for_history'] != 0:
                        raise RuntimeError('GT-history leakage in predicted-history evaluation')
            completed[mode] = str(destination / 'METRICS.json')
            update(completed=completed)
        atomic_json(output / 'BASELINE_REPORT.json', {'protocols': completed, 'sota_claimed': False,
                    'source_report_comparability': 'pending original sampled/preprocessed subset confirmation',
                    'training_started': False})
        update(status='PUBLIC_BASELINES_COMPLETE', success=True, finished_at_unix=time.time())
    except Exception as error:
        update(status='FAILED_NEEDS_ATTENTION', error_type=type(error).__name__, error=str(error), success=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/public_baselines'))
    parser.add_argument('--python', default=str(base / 'envs/qwen_process_seg/bin/python'))
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('GPU count must be positive')
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / 'LAUNCH.json'
        if marker.exists():
            raise RuntimeError('baseline launch already exists; inspect before relaunch')
        command = [sys.executable, '-m', 'projects.evoseg.dialog.run_baselines',
                   '--output', args.output, '--python', args.python, '--gpus', str(args.gpus)]
        with (output / 'driver.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'created_at_unix': time.time()})
        print(json.dumps({'pid': process.pid, 'training_started': False}), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()

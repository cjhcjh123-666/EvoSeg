"""Queue one train-only mechanism diagnostic after the owned public baselines."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from projects.evoseg.restart.overnight import available_gpu_indices
from projects.evoseg.restart.prepare_foundation import atomic_json


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'PROBE.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'status': 'WAITING_FOR_BASELINES', 'new_method_training_started': False}
    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'QUEUE_STATUS.json', state)
    while True:
        path = Path(args.baselines) / 'RUN_STATUS.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['status'] == 'FAILED_NEEDS_ATTENTION':
                update(status='BLOCKED_BY_BASELINE_FAILURE', reason=previous.get('error'))
                return
            if previous['status'] == 'PUBLIC_HALLUSEG_BASELINES_COMPLETE' and previous['success']:
                break
        update()
        time.sleep(20)
    while True:
        devices = available_gpu_indices(1)
        if devices:
            break
        update(status='WAITING_FOR_FREE_GPU')
        time.sleep(20)
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[0]), OMP_NUM_THREADS='4',
               PYTHONPATH=str(root), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    command = [args.python, '-m', 'projects.evoseg.hallucination.probe_interaction', '--output', args.output]
    with (output / 'PROBE.log').open('a') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while child.poll() is None:
            update(status='PROBING_PUBLIC_TRAIN_INTERACTION', child_pid=child.pid, gpu=devices[0])
            time.sleep(15)
    update(status='PROBE_COMPLETE' if child.returncode == 0 else 'PROBE_FAILED',
           child_pid=None, exit_code=child.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--python', default=str(base / 'envs/qwen_process_seg/bin/python'))
    p.add_argument('--baselines', default=str(base / 'results/evoseg_hallucination_20261007/public_baselines'))
    p.add_argument('--output', default=str(base / 'results/evoseg_hallucination_20261007/interaction_probe'))
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / 'LAUNCH.json'
        if marker.exists():
            raise RuntimeError('probe already queued')
        command = [args.python, '-m', 'projects.evoseg.hallucination.run_probe',
                   '--python', args.python, '--baselines', args.baselines, '--output', args.output]
        with (output / 'QUEUE.log').open('a') as log:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {'pid': child.pid, 'command': command})
        print(json.dumps({'queue_pid': child.pid, 'output': args.output}), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()

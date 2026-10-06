"""Audit, small native smoke, then two matched full public baselines on 4+4 GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from projects.evoseg.restart.overnight import available_gpu_indices
from projects.evoseg.restart.prepare_foundation import atomic_json


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'RUN.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    state = {'pid': os.getpid(), 'status': 'PREPARING', 'started_at_unix': time.time(),
             'training_started': False, 'new_method_evaluated': False, 'sota_claimed': False, 'success': False}
    children, logs = [], []

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'RUN_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def launch(name, command, devices=None):
        child_env = dict(env)
        if devices is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        log = (output / (name + '.log')).open('a')
        logs.append(log)
        process = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    def wait(jobs, stage):
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError(stage + ' failed; no full evaluation launched past a failed gate')
            update(status=stage, child_pids=[p.pid for p in jobs if p.poll() is None])
            time.sleep(10)
        if any(p.returncode for p in jobs):
            raise RuntimeError(stage + ' failed')

    models = {'released_sa2va4b': args.base_model, 'faithful4b': args.faithful_model}
    try:
        wait([launch('audit', [args.python, '-m', 'projects.evoseg.hallucination.audit_public',
                    '--output', str(output / 'PUBLIC_DATA_AUDIT.json')])], 'AUDITING_PUBLIC_DATA')
        while True:
            devices = available_gpu_indices(8)
            if devices:
                break
            update(status='WAITING_FOR_EIGHT_FREE_GPUS')
            time.sleep(15)
        update(gpu_indices=devices)
        smoke_jobs = []
        for index, (variant, model) in enumerate(models.items()):
            dest = output / variant / 'smoke'
            command = [args.python, '-m', 'projects.evoseg.hallucination.eval_halluseg',
                       '--model-dir', model, '--output', str(dest), '--limit', '2']
            smoke_jobs.append(launch(variant + '_smoke', command, devices[index * 4:(index + 1) * 4]))
        wait(smoke_jobs, 'VERIFYING_NATIVE_QUARTETS')
        for index, (variant, model) in enumerate(models.items()):
            command = [args.python, '-m', 'projects.evoseg.hallucination.eval_halluseg',
                       '--model-dir', model, '--output', str(output / variant / 'smoke'),
                       '--limit', '2', '--summarize']
            wait([launch(variant + '_smoke_summary', command)], 'SUMMARIZING_NATIVE_SMOKE')
        jobs = []
        for index, (variant, model) in enumerate(models.items()):
            command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                       '-m', 'projects.evoseg.hallucination.eval_halluseg', '--model-dir', model,
                       '--output', str(output / variant / 'full')]
            jobs.append(launch(variant + '_full', command, devices[index * 4:(index + 1) * 4]))
        wait(jobs, 'EVALUATING_FULL_PUBLIC_HALLUSEGBENCH')
        reports = {}
        for variant, model in models.items():
            destination = output / variant / 'full'
            command = [args.python, '-m', 'projects.evoseg.hallucination.eval_halluseg',
                       '--model-dir', model, '--output', str(destination), '--summarize']
            wait([launch(variant + '_full_summary', command)], 'SUMMARIZING_FULL_PUBLIC_RESULTS')
            report = json.loads((destination / 'METRICS.json').read_text())
            if not report['full_public_evaluation'] or report['predictions'] != 6956:
                raise RuntimeError('full public four-combination coverage incomplete')
            reports[variant] = str(destination / 'METRICS.json')
        atomic_json(output / 'BASELINE_REPORT.json', {'status': 'PUBLIC_HALLUSEG_BASELINES_COMPLETE',
            'reports': reports, 'new_method_evaluated': False, 'training_started': False, 'sota_claimed': False})
        update(status='PUBLIC_HALLUSEG_BASELINES_COMPLETE', success=True, child_pids=[],
               finished_at_unix=time.time())
    except Exception as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        update(status='FAILED_NEEDS_ATTENTION', error=str(error), child_pids=[])
        raise
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--python', default=str(base / 'envs/qwen_process_seg/bin/python'))
    p.add_argument('--base-model', default=str(base / 'models/Sa2VA-Qwen3-VL-4B'))
    p.add_argument('--faithful-model', default=str(base / 'models/EvoSeg-Qwen3-VL-4B-Faithful'))
    p.add_argument('--output', default=str(base / 'results/evoseg_hallucination_20261007/public_baselines'))
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / 'LAUNCH.json'
        if marker.exists():
            raise RuntimeError('baseline chain already launched; inspect rather than duplicating')
        command = [args.python, '-m', 'projects.evoseg.hallucination.run_public_baselines',
                   '--python', args.python, '--output', args.output,
                   '--base-model', args.base_model, '--faithful-model', args.faithful_model]
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'training_started': False})
        print(json.dumps({'driver_pid': process.pid, 'output': args.output}), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()

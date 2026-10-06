"""Detached bounded matched pilots and image-disjoint train diagnostics."""
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
    lock = (output / 'PILOTS.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'status': 'PREPARING', 'started_at_unix': time.time(),
             'success': False, 'full_sccs_inference': False, 'sota_claimed': False}
    env = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    jobs, logs = [], []

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'PILOT_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def wait_gpus():
        while True:
            devices = available_gpu_indices(8)
            if devices:
                return devices
            update(status='WAITING_FOR_EIGHT_FREE_GPUS')
            time.sleep(15)

    def launch(name, command, devices):
        child_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
        log = (output / f'{name}.log').open('a')
        logs.append(log)
        child = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append(child)
        return child

    def wait(children, stage):
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError(f'{stage} failed; inspect logs, no continuation authorized')
            update(status=stage, child_pids=[c.pid for c in children if c.poll() is None])
            time.sleep(15)
        if any(child.returncode for child in children):
            raise RuntimeError(f'{stage} failed')

    try:
        resource_lock = None
        if args.resource_lock:
            resource_lock = Path(args.resource_lock).open('a')
            update(status='WAITING_FOR_EXPERIMENT_GPU_LOCK')
            fcntl.flock(resource_lock.fileno(), fcntl.LOCK_EX)
        devices = wait_gpus()
        checkpoint_root = Path(args.checkpoint_root) if args.evaluation_only else output
        if not args.evaluation_only:
            children = []
            for i, variant in enumerate(('plain_lora', 'witness_scope_aux')):
                command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                           '-m', 'projects.evoseg.dialog.train_public_pilot', '--variant', variant,
                           '--output', str(output / variant), '--updates', str(args.updates), '--cache', args.cache]
                if args.initialize_root:
                    command += ['--initialize-from', str(Path(args.initialize_root) / variant)]
                children.append(launch(variant, command, devices[i * 4:(i + 1) * 4]))
            update(status='MATCHED_TRAINING', gpu_indices=devices, training_started=True)
            wait(children, 'MATCHED_TRAINING')
        for rank in range(4):
            traces = [json.loads((checkpoint_root / v / f'STATUS_rank{rank}.json').read_text())['sample_trace_sha256']
                      for v in ('plain_lora', 'witness_scope_aux')]
            if len(set(traces)) != 1:
                raise RuntimeError('matched experiments used different training samples')
        reports = {}
        for variant in ('released_native', 'plain_lora', 'witness_scope_aux'):
            destination = output / 'holdout' / variant
            arguments = ['--history-mode', args.history_mode, '--output', str(destination),
                         '--train-cache-holdout', args.cache, '--dialogue-root', args.train_root]
            if variant != 'released_native':
                arguments += ['--adapter-dir', str(checkpoint_root / variant / 'adapter')]
            devices = wait_gpus()
            command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                       '-m', 'projects.evoseg.dialog.eval_multiturn', *arguments]
            wait([launch(f'eval_{variant}', command, devices)], f'EVALUATING_{variant}')
            command = [args.python, '-m', 'projects.evoseg.dialog.eval_multiturn', *arguments, '--summarize']
            wait([launch(f'summary_{variant}', command, devices)], f'SUMMARIZING_{variant}')
            reports[variant] = json.loads((destination / 'METRICS.json').read_text())
        def aggregate(report):
            values = [r for d in report['datasets'].values() for r in d['rounds'].values()]
            multi = [r for d in report['datasets'].values() for k, r in d['rounds'].items() if int(k) >= 2]
            return {'rounds': sum(r['count'] for r in values),
                    'invalid_outputs': sum(r['invalid_outputs'] for r in values),
                    'multi_round_count': sum(r['count'] for r in multi),
                    'multi_round_gIoU': sum(r['iou_sum'] for r in multi) / sum(r['count'] for r in multi),
                    'multi_round_pooled_cIoU': sum(r['intersection'] for r in multi) / sum(r['union'] for r in multi),
                    'round_counts': {k: sum(d['rounds'].get(k, {}).get('count', 0) for d in report['datasets'].values())
                                     for k in report['mean_across_datasets_by_round']},
                    'mean_round2to6_dataset_cIoU': sum(report['mean_across_datasets_by_round'][str(i)]['mean_dataset_cIoU']
                                                    for i in range(2, 7)) / 5}
        scores = {v: aggregate(r) for v, r in reports.items()}
        if len({s['rounds'] for s in scores.values()}) != 1:
            raise RuntimeError('unequal diagnostic coverage')
        # This bounded pilot never launches larger training merely from lower loss.
        result = {'status': 'MATCHED_PUBLIC_PILOTS_COMPLETE', 'scores': scores,
                  'history_mode': args.history_mode, 'checkpoint_root': str(checkpoint_root),
                  'diagnostic_only': True, 'official_validation_used_for_selection': False,
                  'editing_branch_supervised': False, 'full_sccs_inference': False,
                  'sota_claimed': False, 'larger_training_started': False}
        atomic_json(output / 'PILOT_REPORT.json', result)
        update(status=result['status'], success=True, child_pids=[], scores=scores)
    except Exception as exc:
        # Terminate ONLY this driver's surviving children, never foreign GPU jobs.
        for child in jobs:
            if child.poll() is None:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
        update(status='FAILED', error=str(exc), child_pids=[])
        raise
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--python', default=str(base / 'envs/qwen_process_seg/bin/python'))
    p.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/public_pilots_v1'))
    p.add_argument('--cache', default=str(base / 'results/evoseg_dialog_20261006/public_train_pilot_cache'))
    p.add_argument('--train-root', default=str(base / 'datasets/SegLLM-official/conversations_folder/all_data_mix_train'))
    p.add_argument('--updates', type=int, default=64)
    p.add_argument('--history-mode', choices=['gt_history', 'predicted_history'], default='predicted_history')
    p.add_argument('--evaluation-only', action='store_true', help='Evaluate existing matched adapters; no new updates')
    p.add_argument('--checkpoint-root')
    p.add_argument('--initialize-root', help='Weight/optimizer continuation roots for both matched variants')
    p.add_argument('--resource-lock', help='Serialize project training/evaluation chains without GPU holders')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.evaluation_only and not args.checkpoint_root:
        p.error('evaluation-only requires an explicit checkpoint root')
    if args.evaluation_only and args.initialize_root:
        p.error('evaluation-only cannot initialize training')
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / 'LAUNCH.json'
        if marker.exists():
            raise RuntimeError('pilot already launched; inspect its status instead of duplicating')
        command = [args.python, '-m', 'projects.evoseg.dialog.run_public_pilots',
                   '--output', args.output, '--updates', str(args.updates), '--cache', args.cache,
                   '--train-root', args.train_root, '--history-mode', args.history_mode]
        if args.evaluation_only:
            command += ['--evaluation-only', '--checkpoint-root', args.checkpoint_root]
        if args.initialize_root:
            command += ['--initialize-root', args.initialize_root]
        if args.resource_lock:
            command += ['--resource-lock', args.resource_lock]
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'training_started': False})
        print(json.dumps({'driver_pid': process.pid, 'output': args.output}), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()

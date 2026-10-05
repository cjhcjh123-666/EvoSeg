"""One durable release -> baseline -> train -> evaluate execution chain.

Fails closed on missing weights, incomplete metrics, insufficient free GPUs,
child errors, or an expired training window. Never terminates foreign jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .prepare_foundation import atomic_json


def next_morning_deadline(now=None):
    current = now or datetime.now(ZoneInfo('Asia/Shanghai'))
    deadline = current.replace(hour=9, minute=0, second=0, microsecond=0)
    if deadline <= current:
        deadline += timedelta(days=1)
    return deadline.timestamp()


def available_gpu_indices(required=8, min_free_mib=76000):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    available = []
    for line in result.stdout.splitlines():
        index, free = [int(value.strip()) for value in line.split(',')]
        if free >= min_free_mib:
            available.append(index)
    if len(available) < required:
        return None
    return available[:required]


def prepare_holdout(train_root, run_dir):
    from .native_data import split_video_ids
    videos = json.loads((Path(train_root) / 'meta_expressions_v2.json').read_text())['videos']
    train, held = split_video_ids(videos)
    # Stable, small diagnostic on held-out train videos. Never a benchmark score.
    keys = []
    for video in held[:8]:
        expressions = videos[video]['expressions']
        present = [key for key in sorted(expressions) if expressions[key]['anno_id']]
        absent = [key for key in sorted(expressions) if not expressions[key]['anno_id']]
        keys.extend([[video, key] for key in (present[:2] + absent[:1])])
    if not keys:
        raise RuntimeError('no held-out diagnostic expressions')
    atomic_json(run_dir / 'HELD_OUT_KEYS.json', keys)
    atomic_json(run_dir / 'TRAIN_SPLIT.json', {'train_videos': train, 'held_out_videos': held,
        'diagnostic_expressions': len(keys), 'pretraining_unseen': False})
    return run_dir / 'HELD_OUT_KEYS.json'


def execute(args):
    root, run_dir = Path(__file__).resolve().parents[3], Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'OVERNIGHT.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_FOR_ASSETS', 'pid': os.getpid(), 'training_started': False,
             'deadline_unix': args.deadline, 'deadline_beijing': datetime.fromtimestamp(
                 args.deadline, ZoneInfo('Asia/Shanghai')).isoformat(), 'started_at_unix': time.time(),
             'foundation': 'SaSaSa2VA-26B with original SAM2', 'success': False}
    status_path = run_dir / 'OVERNIGHT_STATUS.json'
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{args.overlay}:{root}', HF_MODULES_CACHE=str(run_dir / 'hf_modules_cache'),
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               MPLCONFIGDIR=str(run_dir / 'matplotlib_cache'), PYTHONUNBUFFERED='1')

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(status_path, state)
        print(json.dumps(state), flush=True)

    def wait_for_gpus():
        while True:
            indices = available_gpu_indices(args.gpus)
            if indices is not None:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in indices)
                return indices
            if time.time() > args.deadline - 2 * 3600:
                raise RuntimeError('insufficient free GPUs before the reserved evaluation window; no foreign jobs touched')
            update(status='WAITING_FOR_FREE_GPUS', requested_gpus=args.gpus)
            time.sleep(30)

    def child(stage, command):
        stage_log = run_dir / f'{stage}.log'
        update(status=stage, command=command, stage_log=str(stage_log))
        with stage_log.open('a') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=process.pid)
            while process.poll() is None:
                # One 15-second heartbeat. A slow stage is not silently called
                # completed; its real process can be inspected independently.
                time.sleep(15)
                detail = {}
                training_status = run_dir / 'native_finetune/TRAINING_STATUS.json'
                if stage in ('NATIVE_PILOT', 'NATIVE_FINETUNE') and training_status.exists():
                    training = json.loads(training_status.read_text())
                    detail = {'training_started': training['training_started'],
                              'training_step': training.get('step', 0),
                              'training_seconds_per_update': training.get('seconds_per_update')}
                update(child_pid=process.pid, **detail)
            if process.returncode:
                raise RuntimeError(f'{stage} failed with exit {process.returncode}; see {stage_log}')
        update(child_pid=None)

    def evaluate(name, split_root, keys=None, checkpoint=None):
        output = run_dir / name
        meta = Path(split_root) / 'meta_expressions_v2.json'
        command = [args.python, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={args.gpus}', '-m', 'projects.evoseg.restart.native_eval',
                   '--model-dir', args.model_dir, '--assets', args.assets, '--images', str(Path(split_root) / 'JPEGImages'),
                   '--meta', str(meta), '--output', str(output)]
        if keys:
            command.extend(['--keys', str(keys)])
        if checkpoint:
            command.extend(['--checkpoint', str(checkpoint)])
        child(name, command)
        from .native_eval import merge_predictions
        key_list = json.loads(Path(keys).read_text()) if keys else None
        count = merge_predictions(meta, output, key_list)
        child(name + '_METRICS', [args.python, '-m', 'projects.evoseg.eval.eval_mevis_jf',
                                 str(output / 'results.json'), '--meta', str(meta),
                                 '--mask', str(Path(split_root) / 'mask_dict.json'),
                                 '--workers', '8', '--output', str(output / 'metrics.json')])
        metrics = json.loads((output / 'metrics.json').read_text())
        if metrics['evaluated_pairs'] != count or metrics['j_and_f'] is None:
            raise RuntimeError(f'{name} has incomplete evaluation coverage')
        return metrics

    try:
        update()
        while True:
            assets = json.loads(Path(args.assets).read_text())
            if assets['status'] == 'ASSETS_READY':
                break
            if assets['status'] == 'DOWNLOAD_FAILED':
                raise RuntimeError(f'foundation download failed: {assets.get("error")}')
            if time.time() > args.deadline - 4 * 3600:
                raise RuntimeError('download did not finish before conservative overnight training window')
            update(downloaded_weight_bytes=assets.get('downloaded_weight_bytes_including_partials', 0),
                   expected_weight_bytes=assets.get('expected_weight_bytes'))
            time.sleep(30)
        gpu_indices = wait_for_gpus()
        update(gpu_indices=gpu_indices)
        keys = prepare_holdout(args.train_root, run_dir)
        # Strict loading is enforced separately by every worker. The first
        # held-out native inference checks the full released path before train.
        baseline_held = evaluate('baseline_heldout', args.train_root, keys)
        baseline = evaluate('baseline_mevis_v2', args.valid_root)
        if baseline['evaluated_pairs'] != 907:
            raise RuntimeError('MeViS-v2 main baseline must cover all 907 expressions')
        stop_at = args.deadline - 2 * 3600
        if time.time() >= stop_at:
            raise RuntimeError('baseline consumed the training window; no shortened result mislabeled as full training')
        wait_for_gpus()
        training_output = run_dir / 'native_finetune'
        train_command = [args.python, '-m', 'torch.distributed.run', '--standalone',
                         f'--nproc_per_node={args.gpus}', '-m', 'projects.evoseg.restart.native_train',
                         '--model-dir', args.model_dir, '--assets', args.assets,
                         '--train-root', args.train_root, '--output', str(training_output),
                         '--max-updates', str(args.max_updates), '--stop-at', str(stop_at)]
        update(training_started=False, baseline=baseline, baseline_heldout=baseline_held)
        child('NATIVE_PILOT', train_command + ['--stop-updates', str(min(20, args.max_updates))])
        pilot_checkpoint = json.loads((training_output / 'LATEST.json').read_text())['checkpoint']
        update(training_started=True, pilot_checkpoint=pilot_checkpoint)
        pilot = evaluate('pilot_heldout', args.train_root, keys, pilot_checkpoint)
        if pilot['target_present']['j_and_f'] is None or baseline_held['target_present']['j_and_f'] is None:
            raise RuntimeError('pilot has no valid target-present diagnostic')
        pilot_delta = pilot['target_present']['j_and_f'] - baseline_held['target_present']['j_and_f']
        update(pilot_target_present_delta_points=100 * pilot_delta)
        if pilot_delta < -.03:
            raise RuntimeError('pilot lost more than 3 J&F points on held-out target-present expressions; full training withheld')
        wait_for_gpus()
        child('NATIVE_FINETUNE', train_command + ['--resume', str(pilot_checkpoint)])
        training = json.loads((training_output / 'TRAINING_STATUS.json').read_text())
        if training['status'] != 'FIRST_RUN_COMPLETE' or training['step'] < 1:
            raise RuntimeError('training did not finish and save a valid first-run checkpoint')
        checkpoint = json.loads((training_output / 'LATEST.json').read_text())['checkpoint']
        update(training_started=True, training=training, checkpoint=checkpoint)
        held = evaluate('finetune_heldout', args.train_root, keys, checkpoint)
        fine = evaluate('finetune_mevis_v2', args.valid_root, checkpoint=checkpoint)
        if fine['evaluated_pairs'] != 907:
            raise RuntimeError('fine-tuned main evaluation must cover all 907 expressions')
        summary = {'foundation': state['foundation'], 'foundation_revision': assets['revision'],
                   'training': training, 'checkpoint': checkpoint,
                   'baseline': baseline, 'finetuned': fine, 'heldout_before': baseline_held,
                   'heldout_after': held, 'delta_jf_points': 100 * (fine['j_and_f'] - baseline['j_and_f']),
                   'sota_claimed': False, 'met_deadline': time.time() <= args.deadline}
        atomic_json(run_dir / 'FIRST_RUN_REPORT.json', summary)
        update(status='FIRST_RUN_EVALUATED', success=True, report=str(run_dir / 'FIRST_RUN_REPORT.json'),
               met_deadline=summary['met_deadline'], finished_at_unix=time.time())
    except Exception as error:
        update(status='FAILED_NEEDS_ATTENTION', success=False, error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--run-dir', default=str(base / 'results/evoseg_restart_20261006'))
    parser.add_argument('--model-dir', default=str(base / 'models/SaSaSa2VA-26B-official'))
    parser.add_argument('--assets', default=str(base / 'results/evoseg_restart_20261006/ASSETS.json'))
    parser.add_argument('--train-root', default=str(base / 'datasets/mevis_v2/train'))
    parser.add_argument('--valid-root', default=str(base / 'datasets/mevis_v2/valid_u'))
    parser.add_argument('--python', default=str(base / 'envs/virst/bin/python'))
    parser.add_argument('--overlay', default=str(base / 'envs/sasasa2va_native_overlay'))
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--max-updates', type=int, default=200)
    parser.add_argument('--deadline', type=float, default=next_morning_deadline())
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.gpus < 1 or args.max_updates < 1:
        parser.error('GPU and update counts must be positive')
    if args.detach:
        root, run_dir = Path(__file__).resolve().parents[3], Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / 'OVERNIGHT_LAUNCH.json'
        if marker.exists():
            raise RuntimeError('overnight launch already exists; inspect process before relaunching')
        command = [sys.executable, '-m', 'projects.evoseg.restart.overnight']
        for key, value in vars(args).items():
            if key != 'detach':
                command.extend(['--' + key.replace('_', '-'), str(value)])
        with (run_dir / 'overnight.log').open('a') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'created_at_unix': time.time()})
        print(json.dumps({'pid': process.pid, 'deadline': args.deadline, 'training_started': False}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

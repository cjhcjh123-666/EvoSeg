"""Resume saved public evaluations serially; no retraining or public model selection."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from projects.evoseg.restart.prepare_foundation import atomic_json
from .continue_training import check_relaunch


def free_devices(min_free_mib=76000):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    available = [int(line.split(',')[0]) for line in result.stdout.splitlines()
                 if int(line.split(',')[1]) >= min_free_mib]
    for size in (8, 4, 2):
        if len(available) >= size:
            return available[:size]
    return None


def low_headroom(devices, minimum_mib=16384):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in result.stdout.splitlines()}
    return [device for device in devices if free.get(device, 0) < minimum_mib]


def foreign_gpu_processes(destination, devices):
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)
    selected = {line.split(',')[1].strip() for line in inventory.stdout.splitlines()
                if int(line.split(',')[0]) in devices}
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    foreign = []
    for line in result.stdout.splitlines():
        uuid, pid, memory = [v.strip() for v in line.split(',')]
        pid, memory = int(pid), int(memory)
        if uuid not in selected:
            continue
        if memory < 1024:
            continue
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError:
            foreign.append(pid)
            continue
        if str(destination).encode() not in command:
            foreign.append(pid)
    return sorted(set(foreign))


def execute(args):
    root = Path(__file__).resolve().parents[3]
    base = Path('/9950backfile/chenjiahui/evo_artifacts/results/evoseg_hallucination_20261007')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'RECOVERY.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    shared = getattr(args, 'shared_gpus', False)
    source_cache = base / 'reasoning_public_mix/cache'
    selection = json.loads((base / 'extended_recovery_run/SELECTION.json').read_text())
    regional = json.loads((base / 'region_public_pilot/REGION_STATUS.json').read_text())['selected']
    if not all(v['qualified'] for v in regional['guards'].values()):
        raise RuntimeError('regional pair was not selected by the private preservation guard')
    tasks = []
    # Finish the interrupted control first. Next prioritize the two full
    # regional/global HalluSegBench runs, not additional training or data changes.
    for variant in ('interaction', 'full_view'):
        tasks.append((variant + '_halluseg', 'halluseg', selection['checkpoints'][variant],
                      base / 'extended_recovery_run/evaluation' / (variant + '_halluseg')))
    for mode in ('regional', 'global'):
        tasks.append((mode + '_halluseg', 'halluseg', regional['checkpoints'][mode],
                      base / 'region_public_pilot/evaluation' / (mode + '_halluseg')))
    for family, pairs, directory in [('proposal', selection['checkpoints'], 'extended_recovery_run'),
                                     ('region', regional['checkpoints'], 'region_public_pilot')]:
        for name, checkpoint in pairs.items():
            tasks.append((family + '_' + name + '_gref_val', 'gref_val', checkpoint,
                          base / directory / 'evaluation' / (name + '_gref_val')))
    env = dict(os.environ, PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', MASK_TOKENIZER_NUM_MASK_TOKEN='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if shared:
        env['EVOSEG_EVAL_GPU_MEMORY_FRACTION'] = '.45'
    state = {'pid': os.getpid(), 'deadline': args.deadline, 'status': 'STARTING',
             'serial_gpu_jobs': True, 'retraining': False, 'weights_changed': False,
             'selection_uses_public_benchmark': False, 'success': False, 'sota_claimed': False}
    state.update(shared_gpus=shared, minimum_start_free_mib=48000 if shared else 76000,
                 torch_memory_fraction=.45 if shared else None)
    finished = {}
    active = None

    def update(**values):
        state.update(values, updated_at_unix=time.time(), completed_tasks=list(finished))
        atomic_json(output / 'RECOVERY_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def stop_owned(process):
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)

    try:
        update()
        for name, benchmark, checkpoint, destination in tasks:
            expected = 1739 if benchmark == 'halluseg' else 14229
            metrics = destination / 'METRICS.json'
            if metrics.exists():
                existing = json.loads(metrics.read_text())
                if (existing['cases'] == expected and existing['full_public_evaluation'] and
                        existing['checkpoint'] == checkpoint and
                        len(list((destination / 'cases').glob('*.json'))) == expected):
                    finished[name] = existing
                    update(status='REUSED_VERIFIED_COMPLETE_EVALUATION', stage=name)
                    continue
            arguments = ['--cache', str(source_cache), '--output', str(destination),
                         '--benchmark', benchmark, '--checkpoint', checkpoint]
            attempts = 0
            while True:
                if time.time() >= deadline:
                    raise TimeoutError('user deadline reached; incomplete predictions are preserved')
                devices = free_devices(48000 if shared else 76000)
                if not devices:
                    update(status='WAITING_FOR_FREE_GPUS', stage=name, child_pid=None)
                    time.sleep(20)
                    continue
                attempts += 1
                child_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
                command = [args.python, '-m', 'torch.distributed.run', '--standalone',
                           f'--nproc_per_node={len(devices)}', '-m', 'projects.evoseg.hallucination.eval_fresh', *arguments]
                conflict = False
                with (output / f'{name}_attempt_{attempts}.log').open('a') as log:
                    active = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                    update(status='EVALUATING', stage=name, child_pid=active.pid, devices=devices,
                           prediction_directory=str(destination), attempt=attempts)
                    while active.poll() is None:
                        if time.time() >= deadline:
                            stop_owned(active)
                            raise TimeoutError('deadline reached during ' + name)
                        other = foreign_gpu_processes(destination, devices)
                        pressure = low_headroom(devices) if shared else []
                        if (other and not shared) or pressure:
                            # Only our own worker process group is terminated.
                            # Never pause, kill or modify another GPU task.
                            stop_owned(active)
                            update(status='DEFERRED_GPU_CONFLICT', foreign_pids=other,
                                   low_headroom_devices=pressure, child_pid=None)
                            conflict = True
                            break
                        update(status='EVALUATING', stage=name, child_pid=active.pid,
                               concurrent_gpu_pids=other if shared else [],
                               case_files=len(list((destination / 'cases').glob('*.json'))))
                        time.sleep(15)
                if conflict:
                    time.sleep(20)
                    continue
                if active.returncode == 0:
                    break
                if attempts >= 3:
                    raise RuntimeError(name + ' failed three times; inspect recovery logs')
                update(status='RETRYING_SAVED_PREDICTIONS', stage=name, child_pid=None)
                time.sleep(20)
            with (output / (name + '_summary.log')).open('a') as log:
                active = subprocess.Popen([args.python, '-m', 'projects.evoseg.hallucination.eval_fresh',
                                           *arguments, '--summarize'], cwd=root, env=env,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while active.poll() is None:
                    if time.time() >= deadline:
                        stop_owned(active)
                        raise TimeoutError('deadline reached during summary')
                    update(status='SUMMARIZING', stage=name, child_pid=active.pid)
                    time.sleep(15)
                if active.returncode:
                    raise RuntimeError(name + ' summary failed')
            result = json.loads(metrics.read_text())
            if result['cases'] != expected or not result['full_public_evaluation']:
                raise RuntimeError('incomplete public split: ' + name)
            finished[name] = result
            update(status='TASK_COMPLETE', stage=name, child_pid=None)
        atomic_json(output / 'RECOVERY_REPORT.json', {'results': finished, 'expected_tasks': len(tasks),
            'full_evaluation_complete': len(finished) == len(tasks), 'retraining': False,
            'weights_changed': False, 'sota_claimed': False})
        update(status='ALL_PUBLIC_EVALUATIONS_COMPLETE', success=True, child_pid=None)
    except Exception as error:
        stop_owned(active)
        update(status='DEADLINE_REACHED_PARTIAL_RESULTS' if isinstance(error, TimeoutError) else 'FAILED_PREDICTIONS_PRESERVED',
               error=str(error), child_pid=None)
        atomic_json(output / 'RECOVERY_REPORT.json', {'results': finished, 'expected_tasks': len(tasks),
            'full_evaluation_complete': False, 'error': str(error), 'retraining': False, 'sota_claimed': False})
        if not isinstance(error, TimeoutError):
            raise


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=base + '/results/evoseg_hallucination_20261007/overnight_evaluation_recovery')
    p.add_argument('--python', default=base + '/envs/qwen_process_seg/bin/python')
    p.add_argument('--deadline', default='2026-10-08T09:00:00+08:00')
    p.add_argument('--detach', action='store_true')
    p.add_argument('--resume-supervision', action='store_true')
    p.add_argument('--shared-gpus', action='store_true', help='user-authorized memory-capped sharing; never signal other tasks')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        check_relaunch(output, args.resume_supervision)
        command = [args.python, '-m', 'projects.evoseg.hallucination.recover_evaluations',
                   '--output', args.output, '--deadline', args.deadline]
        if args.shared_gpus:
            command += ['--shared-gpus']
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': process.pid, 'command': command, 'deadline': args.deadline})
        atomic_json(output / 'RECOVERY_STATUS.json', {'pid': process.pid, 'status': 'SUPERVISOR_STARTING',
                                                   'deadline': args.deadline})
        print(json.dumps({'recovery_pid': process.pid, 'deadline': args.deadline}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

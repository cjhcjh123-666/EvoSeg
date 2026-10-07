"""Detached fixed-budget regional/global pilot after the existing public queue."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from projects.evoseg.restart.overnight import available_gpu_indices
from projects.evoseg.restart.prepare_foundation import atomic_json
from .continue_training import preservation_guard, check_relaunch
from .train_fresh import cached_records


def execute(args):
    root, output = Path(__file__).resolve().parents[3], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'REGION.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    env = dict(os.environ, PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
               HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', MASK_TOKENIZER_NUM_MASK_TOKEN='1')
    state = {'pid': os.getpid(), 'deadline': args.deadline, 'language_frozen': True,
             'generated_training_data': False, 'success': False, 'sota_claimed': False}
    children, logs = [], []

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'REGION_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def GPUs():
        while time.time() < deadline:
            devices = available_gpu_indices(8)
            if devices:
                return devices
            update(status='WAITING_FOR_FREE_GPUS')
            time.sleep(20)
        raise RuntimeError('deadline reached waiting for genuinely free GPUs')

    def readonly_shared_devices():
        """Share only with our long public inference, never foreign or LoRA jobs.

        The VLM cache needs 55GB free on every GPU. The subsequent SAM-only
        head phase uses the same indices and its measured footprint is tiny.
        """
        prior = Path(args.after_run)
        while time.time() < deadline - 3 * 3600:
            if (prior / 'EXTENDED_REPORT.json').exists():
                return GPUs()
            status = json.loads((prior / 'EXTENDED_STATUS.json').read_text()) if (prior / 'EXTENDED_STATUS.json').exists() else {}
            if status.get('status', '').startswith('FAILED'):
                raise RuntimeError('prior public queue failed: ' + status.get('error', 'unknown'))
            stage = status.get('status', '')
            if stage.startswith('EVALUATING_') and ('halluseg' in stage or 'gref_val' in stage):
                available = available_gpu_indices(8, min_free_mib=55000)
                check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, check=True)
                pids = {int(line.strip()) for line in check.stdout.splitlines() if line.strip()}
                own_inference = len(pids) == 8
                for pid in pids:
                    try:
                        command = Path(f'/proc/{pid}/cmdline').read_bytes()
                    except FileNotFoundError:
                        own_inference = False
                        break
                    if (b'projects.evoseg.hallucination.eval_fresh' not in command or
                            b'evoseg_hallucination_20261007' not in command):
                        own_inference = False
                if available and own_inference:
                    update(sharing_only_owned_readonly_inference=True, cache_minimum_free_mib=55000)
                    return available
            update(status='WAITING_FOR_SAFE_READONLY_CACHE_WINDOW')
            time.sleep(20)
        return None

    def launch(name, command, devices=None):
        log = (output / (name + '.log')).open('a')
        logs.append(log)
        child_env = dict(env)
        if devices is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        process = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    def wait(jobs, stage, progress=()):
        signature, active = None, time.monotonic()
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError(stage + ' failed; see its log')
            current = tuple(sorted((str(path), path.stat().st_mtime_ns)
                            for directory in progress for path in directory.glob('STATUS_rank*.json')))
            if current != signature:
                signature, active = current, time.monotonic()
            if time.monotonic() - active > 1200:
                raise RuntimeError(stage + ' stalled for 20 minutes')
            if time.time() >= deadline:
                raise RuntimeError('22:00 deadline reached; preserve checkpoints and partial predictions')
            update(status=stage, child_pids=[p.pid for p in jobs if p.poll() is None])
            time.sleep(15)
        if any(p.returncode for p in jobs):
            raise RuntimeError(stage + ' failed')

    def distributed(module, arguments, devices):
        return [args.python, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={len(devices)}', '-m', module, *arguments]

    def evaluate(name, checkpoint, devices, benchmark='holdout'):
        directory = output / 'evaluation' / name
        arguments = ['--cache', args.source_cache, '--output', str(directory), '--benchmark', benchmark,
                     '--checkpoint', checkpoint]
        wait([launch('eval_' + name, distributed('projects.evoseg.hallucination.eval_fresh', arguments, devices), devices)],
             'EVALUATING_' + name, [directory])
        wait([launch('summary_' + name, [args.python, '-m', 'projects.evoseg.hallucination.eval_fresh',
                                       *arguments, '--summarize'])], 'SUMMARIZING_' + name)
        return json.loads((directory / 'METRICS.json').read_text())

    def cases(name):
        return {p.stem: json.loads(p.read_text())['metrics']
                for p in (output / 'evaluation' / name / 'cases').glob('*.json')}

    def evaluate_private(name, checkpoint, devices):
        directory = output / 'evaluation' / name
        arguments = ['evaluate-cache', '--cache', str(output / 'cache'), '--output', str(directory)]
        if checkpoint:
            arguments += ['--checkpoint', checkpoint]
        wait([launch('eval_' + name, distributed('projects.evoseg.hallucination.region_pilot', arguments, devices), devices)],
             'EVALUATING_' + name, [directory])
        arguments[0] = 'summarize-cache'
        wait([launch('summary_' + name, [args.python, '-m', 'projects.evoseg.hallucination.region_pilot', *arguments])],
             'SUMMARIZING_' + name)
        return json.loads((directory / 'METRICS.json').read_text())

    try:
        update(status='WAITING_FOR_SAFE_READONLY_CACHE_WINDOW')
        prior = Path(args.after_run)
        devices = readonly_shared_devices()
        if devices is None:
            update(status='DEFERRED_NO_SAFE_PILOT_WINDOW', pending_experiment=True)
            return
        cache = output / 'cache'
        arguments = ['cache', '--source-cache', args.source_cache, '--parent', args.parent,
                     '--output', str(cache), '--max-records', '1024']
        wait([launch('native_region_cache', distributed('projects.evoseg.hallucination.region_pilot', arguments, devices), devices)],
             'CACHING_NATIVE_REGION_FEATURES', [cache])
        index = json.loads((cache / 'INDEX.json').read_text())
        if index['smoke_only'] or len(list((cache / 'records').glob('*.json'))) != len(index['records']):
            raise RuntimeError('incomplete or diagnostic-only regional cache')
        # No language model in this training phase; both arms share a frozen
        # parent and exactly the same cached model proposal inputs/human masks.
        jobs, progress = [], []
        for index, mode in enumerate(('global', 'regional')):
            destination = output / 'training' / mode
            arguments = ['train', '--cache', str(cache), '--output', str(destination), '--mode', mode,
                         '--steps', '1000', '--stop-at', str(deadline - 150 * 60)]
            group = devices[index * 4:(index + 1) * 4]
            jobs.append(launch('train_' + mode, distributed('projects.evoseg.hallucination.region_pilot', arguments, group), group))
            progress.append(destination)
        wait(jobs, 'TRAINING_FROZEN_LANGUAGE_REGION_PILOT', progress)
        reference_score = evaluate_private('parent_holdout', None, devices)
        reference = cases('parent_holdout')
        image_ids = {r['id']: r['image_id'] for r in cached_records(args.source_cache) if r['holdout']}
        if len(reference) != len(image_ids):
            raise RuntimeError('private holdout coverage mismatch')
        history, best = [], None
        for step in (100, 300, 1000):
            checkpoints = {mode: str(output / 'training' / mode / f'checkpoint_{step:06d}')
                           for mode in ('global', 'regional')}
            if not all(Path(p).is_dir() for p in checkpoints.values()):
                continue
            scores, guards = {}, {}
            for mode in ('global', 'regional'):
                name = f'{mode}_holdout_{step}'
                scores[mode] = evaluate_private(name, checkpoints[mode], devices)
                guards[mode] = preservation_guard(reference, cases(name), image_ids)
            item = {'step': step, 'checkpoints': checkpoints, 'scores': scores, 'guards': guards}
            history.append(item)
            if all(g['qualified'] for g in guards.values()) and (best is None or
                    scores['regional']['gIoU'] > best['scores']['regional']['gIoU']):
                best = item
            update(private_reference=reference_score, history=history, selected=best)
        public = {}
        if best:
            # Small cached private checks may overlap the existing public job.
            # Full public predictions wait for it and use genuine free GPUs.
            while not (prior / 'EXTENDED_REPORT.json').exists() and time.time() < deadline:
                if (prior / 'EXTENDED_STATUS.json').exists():
                    other = json.loads((prior / 'EXTENDED_STATUS.json').read_text())
                    if other['status'].startswith('FAILED'):
                        raise RuntimeError('prior public evaluation needs recovery')
                update(status='PRIVATE_PILOT_READY_WAITING_FOR_PUBLIC_EVALUATION', selected=best)
                time.sleep(20)
            devices = GPUs()
            for benchmark in ('halluseg', 'gref_val'):
                for mode in ('regional', 'global'):
                    estimate = (55 if benchmark == 'halluseg' else 90) * 60
                    if time.time() + estimate > deadline:
                        update(full_public_evaluation_pending=True)
                        break
                    public[mode + '_' + benchmark] = evaluate(mode + '_' + benchmark,
                        best['checkpoints'][mode], devices, benchmark)
        atomic_json(output / 'REGION_REPORT.json', {**state, 'selected': best, 'history': history,
            'public_results': public, 'full_public_comparison_complete': len(public) == 4,
            'selection_uses_public_benchmark': False, 'language_generation_changed': False,
            'execution_success_not_method_superiority': True})
        update(status='REGION_PILOT_FINISHED', success=True, child_pids=[],
               method_qualified_on_private_holdout=best is not None, full_public_comparison_complete=len(public) == 4)
    except Exception as error:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        update(status='FAILED_CHECKPOINTS_PRESERVED', error=str(error))
        raise
    finally:
        for log in logs:
            log.close()


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    folder = base + '/results/evoseg_hallucination_20261007'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache', default=folder + '/reasoning_public_mix/cache')
    p.add_argument('--parent', default=folder + '/new_public_run/training/interaction/checkpoint_000100')
    p.add_argument('--after-run', default=folder + '/extended_recovery_run')
    p.add_argument('--output', default=folder + '/region_public_pilot')
    p.add_argument('--python', default=base + '/envs/qwen_process_seg/bin/python')
    p.add_argument('--deadline', default='2026-10-07T22:00:00+08:00')
    p.add_argument('--detach', action='store_true')
    p.add_argument('--resume-supervision', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        check_relaunch(output, args.resume_supervision)
        command = [args.python, '-m', 'projects.evoseg.hallucination.run_region_pilot',
                   '--source-cache', args.source_cache, '--parent', args.parent,
                   '--after-run', args.after_run, '--output', args.output, '--deadline', args.deadline]
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': process.pid, 'command': command, 'deadline': args.deadline})
        atomic_json(output / 'REGION_STATUS.json', {'pid': process.pid, 'status': 'SUPERVISOR_STARTING',
                    'deadline': args.deadline, 'updated_at_unix': time.time()})
        print(json.dumps({'regional_queue_pid': process.pid, 'deadline': args.deadline}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

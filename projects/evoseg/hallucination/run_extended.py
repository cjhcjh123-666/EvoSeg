"""Queue pure continuation -> audited reasoning/rollout experiment -> full public eval."""
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
from .continue_training import check_relaunch


def matched_candidate(report):
    """Select a same-step pair from private diagnostics, never public scores."""
    eligible = [v for v in report['history'] if all(g['qualified'] for g in v['guards'].values())]
    if not eligible:
        return None
    choice = max(eligible, key=lambda v: (v['scores']['interaction']['gIoU'], -v['requested_step']))
    return {'checkpoints': choice['checkpoints'], 'selection_step': choice['requested_step'],
            'private_scores': choice['scores'], 'same_steps_and_training_data': True,
            'selection_uses_public_benchmark': False}


def execute(args):
    root, output = Path(__file__).resolve().parents[3], Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'EXTENDED.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    env = dict(os.environ, PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
               HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', MASK_TOKENIZER_NUM_MASK_TOKEN='1')
    children, logs = [], []
    state = {'pid': os.getpid(), 'deadline': args.deadline, 'success': False,
             'generated_training_data': False, 'sota_claimed': False}

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'EXTENDED_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def wait_report(directory, name, status_name):
        path = directory / name
        while not path.exists():
            if time.time() >= deadline:
                raise RuntimeError('deadline reached while waiting for ' + str(path))
            if (directory / status_name).exists():
                other = json.loads((directory / status_name).read_text())
                if other['status'].startswith('FAILED'):
                    raise RuntimeError('prior stage failed: ' + other.get('error', 'unknown'))
            update(status='WAITING_FOR_' + name, prerequisite=str(path))
            time.sleep(20)
        return json.loads(path.read_text())

    def GPUs():
        while time.time() < deadline:
            devices = available_gpu_indices(8)
            if devices:
                return devices
            update(status='WAITING_FOR_FREE_GPUS')
            time.sleep(20)
        raise RuntimeError('deadline reached waiting for free GPUs')

    def child(name, arguments, devices=None):
        log = (output / (name + '.log')).open('a')
        logs.append(log)
        child_env = dict(env)
        if devices is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        process = subprocess.Popen([args.python, *arguments], cwd=root, env=child_env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        update(status=name, child_pid=process.pid)
        while process.poll() is None:
            if time.time() >= deadline:
                raise RuntimeError('deadline reached; retain checkpoint and partial evaluation cache')
            update(status=name, child_pid=process.pid)
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(name + ' failed; see stage log')
        update(child_pid=None)

    try:
        update(status='WAITING_FOR_LONGER_TRAINING')
        longer = wait_report(Path(args.longer_run), 'CONTINUE_REPORT.json', 'CONTINUE_STATUS.json')
        mix = Path(args.reasoning_mix)
        while not (mix / 'SOURCE_AUDIT.json').exists():
            if time.time() >= deadline - 5 * 3600:
                raise RuntimeError('reasoning-source audit did not complete before the reserved evaluation window')
            update(status='WAITING_FOR_PUBLIC_REASONING_SOURCE_AUDIT')
            time.sleep(20)
        if not (mix / 'CACHE_READY.json').exists():
            devices = GPUs()
            child('ENCODING_PUBLISHED_REASONING_GT', ['-m', 'projects.evoseg.hallucination.prepare_reasoning',
                                                   '--output', str(mix), '--encode'], devices[:1])
        ready = json.loads((mix / 'CACHE_READY.json').read_text())
        if ready['records'] != 16623 or ready['generated_queries'] or ready['pseudo_labels']:
            raise RuntimeError('audited original-data cache coverage failed')
        optimized = Path(args.existing_optimized_run) if args.existing_optimized_run else output / 'optimized_training'
        if args.existing_optimized_run:
            report = wait_report(optimized, 'CONTINUE_REPORT.json', 'CONTINUE_STATUS.json')
        else:
            child('REASONING_AND_NATIVE_PROPOSAL_TRAINING', ['-m', 'projects.evoseg.hallucination.continue_training',
            '--source-run', args.source_run, '--output', str(optimized), '--train-cache', str(mix / 'cache'),
            '--deadline', args.deadline, '--reserve-minutes', '300', '--fork-objective',
            '--prompt-rollout-rate', '.25', '--milestones', '20', '100', '300', '600', '1200', '--skip-public'])
            report = json.loads((optimized / 'CONTINUE_REPORT.json').read_text())
        candidate = matched_candidate(report)
        selected_route = 'public_reasoning_native_proposal'
        if candidate is None:
            candidate, selected_route = matched_candidate(longer), 'pure_longer_training'
        if candidate is None:
            previous = json.loads((Path(args.source_run) / 'FIRST_RUN_REPORT.json').read_text())
            candidate = {'checkpoints': previous['checkpoints'], 'selection_step': 100,
                         'same_steps_and_training_data': True, 'fallback_to_preserved_first_run': True}
            selected_route = 'preserved_first_run'
        atomic_json(output / 'SELECTION.json', {**candidate, 'selected_route': selected_route,
                                               'selection_uses_public_benchmark': False})
        update(selected_route=selected_route, selection=candidate)
        public = {}
        for benchmark in ('halluseg', 'gref_val'):
            for variant in ('interaction', 'full_view'):
                estimate = 50 * 60 if benchmark == 'halluseg' else 80 * 60
                if time.time() + estimate > deadline:
                    update(full_public_comparison_pending=True)
                    break
                destination = output / 'evaluation' / (variant + '_' + benchmark)
                arguments = ['-m', 'projects.evoseg.hallucination.eval_fresh', '--cache', str(mix / 'cache'),
                             '--output', str(destination), '--benchmark', benchmark,
                             '--checkpoint', candidate['checkpoints'][variant]]
                devices = GPUs()
                child('EVALUATING_' + variant + '_' + benchmark,
                      ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', *arguments], devices)
                child('SUMMARIZING_' + variant + '_' + benchmark, arguments + ['--summarize'])
                public[variant + '_' + benchmark] = json.loads((destination / 'METRICS.json').read_text())
        atomic_json(output / 'EXTENDED_REPORT.json', {**state, 'public_results': public,
            'full_public_comparison_complete': len(public) == 4,
            'released_reference_report': str(Path(args.source_run) / 'FIRST_RUN_REPORT.json'),
            'selection_uses_public_benchmark': False, 'execution_success_not_sota': True})
        update(status='EXTENDED_CHAIN_FINISHED', success=True, child_pid=None,
               full_public_comparison_complete=len(public) == 4)
    except Exception as error:
        for process in children:
            if process.poll() is None:
                # Child continuation has its own detached worker sessions.
                # Resolve only the worker PIDs explicitly recorded by that
                # owned child; never search/kill unrelated GPU processes.
                nested = output / 'optimized_training/CONTINUE_STATUS.json'
                if nested.exists():
                    owned = json.loads(nested.read_text())
                    if owned.get('pid') == process.pid:
                        for pid in owned.get('child_pids', []):
                            try:
                                os.killpg(pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
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
    p.add_argument('--source-run', default=folder + '/new_public_run')
    p.add_argument('--longer-run', default=folder + '/longer_public_run')
    p.add_argument('--reasoning-mix', default=folder + '/reasoning_public_mix')
    p.add_argument('--output', default=folder + '/extended_public_run')
    p.add_argument('--deadline', default='2026-10-07T22:00:00+08:00')
    p.add_argument('--python', default=base + '/envs/qwen_process_seg/bin/python')
    p.add_argument('--existing-optimized-run', help='wait for a separately recovered owned training chain')
    p.add_argument('--resume-supervision', action='store_true')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        check_relaunch(output, args.resume_supervision)
        command = [args.python, '-m', 'projects.evoseg.hallucination.run_extended',
                   '--source-run', args.source_run, '--longer-run', args.longer_run,
                   '--reasoning-mix', args.reasoning_mix, '--output', args.output, '--deadline', args.deadline]
        if args.existing_optimized_run:
            command += ['--existing-optimized-run', args.existing_optimized_run]
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': process.pid, 'command': command, 'deadline': args.deadline})
        atomic_json(output / 'EXTENDED_STATUS.json', {'pid': process.pid, 'status': 'SUPERVISOR_STARTING',
                    'deadline': args.deadline, 'updated_at_unix': time.time()})
        print(json.dumps({'queue_pid': process.pid, 'deadline': args.deadline}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

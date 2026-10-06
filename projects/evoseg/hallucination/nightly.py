"""Bounded supervised fresh-model training/evaluation until a user-chosen deadline."""
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


def execute(args):
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'NIGHTLY.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    train_stop = deadline - 2 * 3600
    env = dict(os.environ, PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_XET='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', MASK_TOKENIZER_NUM_MASK_TOKEN='1')
    state = {'pid': os.getpid(), 'status': 'PREPARING', 'deadline': args.deadline,
             'deadline_unix': deadline, 'training_stop_unix': train_stop,
             'started_at_unix': time.time(), 'training_started': False, 'success': False,
             'fresh_foundation': 'zhouyik/Qwen3-VL-8B-SAMTok', 'old_faithful_inherited': False,
             'generated_training_data': False, 'sota_claimed': False}
    jobs, logs = [], []

    def update(**values):
        state.update(values, updated_at_unix=time.time(), seconds_to_deadline=max(0, deadline - time.time()))
        atomic_json(output / 'NIGHTLY_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def gpus(number):
        while time.time() < deadline:
            devices = available_gpu_indices(number)
            if devices:
                return devices
            update(status='WAITING_FOR_FREE_GPUS', requested_gpus=number)
            time.sleep(20)
        raise RuntimeError('deadline reached while waiting for free GPUs')

    def launch(name, command, devices=None):
        child_env = dict(env)
        if devices is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        log = (output / (name + '.log')).open('a')
        logs.append(log)
        child = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append(child)
        update(stage=name, child_pids=[p.pid for p in jobs if p.poll() is None])
        return child

    def wait(children, stage):
        last_signature, last_activity = None, time.monotonic()
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(stage + ' failed; inspect stage log')
            update(status=stage, child_pids=[p.pid for p in children if p.poll() is None])
            paths = []
            if stage == 'ENCODING_ORIGINAL_HUMAN_GT':
                paths = list((output / 'cache').glob('STATUS_rank*.json'))
            elif stage == 'MATCHED_COUPLED_TRAINING':
                paths = list((output / 'training').glob('*/STATUS_rank*.json'))
            elif stage.startswith('EVALUATING_'):
                paths = list((output / 'evaluation' / stage.removeprefix('EVALUATING_')).glob('STATUS_rank*.json'))
            signature = []
            for path in sorted(paths):
                values = json.loads(path.read_text())
                signature.append((str(path), values.get('step', values.get('cases_done', values.get('records_done', 0)))))
            signature = tuple(signature)
            if signature != last_signature:
                last_signature, last_activity = signature, time.monotonic()
            if time.monotonic() - last_activity > 1200:
                raise RuntimeError(stage + ' made no measurable progress for 20 minutes')
            if time.time() >= deadline:
                raise RuntimeError('user deadline reached; terminate only owned child jobs')
            time.sleep(15)
        if any(p.returncode for p in children):
            raise RuntimeError(stage + ' failed')

    def distributed(module, arguments, devices):
        return [args.python, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={len(devices)}', '-m', module, *arguments]

    manifest = output / 'TRAIN_MANIFEST.json'
    cache = output / 'cache'

    def evaluate(name, checkpoint=None, limit=None, benchmark='holdout'):
        destination = output / 'evaluation' / name
        arguments = ['--cache', str(cache), '--output', str(destination), '--benchmark', benchmark]
        if checkpoint:
            arguments += ['--checkpoint', checkpoint]
        if limit:
            arguments += ['--limit', str(limit)]
        devices = gpus(8)
        command = distributed('projects.evoseg.hallucination.eval_fresh', arguments, devices)
        wait([launch('eval_' + name, command, devices)], 'EVALUATING_' + name)
        wait([launch('summary_' + name, [args.python, '-m', 'projects.evoseg.hallucination.eval_fresh',
                                       *arguments, '--summarize'])], 'SUMMARIZING_' + name)
        return json.loads((destination / 'METRICS.json').read_text())

    try:
        update()
        if not manifest.exists():
            wait([launch('public_source_verification', [args.python, '-m', 'projects.evoseg.hallucination.prepare_sources'])],
                 'VERIFYING_OFFICIAL_DATA')
            wait([launch('public_manifest', [args.python, '-m', 'projects.evoseg.hallucination.public_mix',
                         '--output', str(manifest), '--max-records', '16384'])], 'AUDITING_PUBLIC_TRAIN_MANIFEST')
        source = json.loads(manifest.read_text())
        if not source['source_only_train'] or source['generated_queries'] or source['pseudo_labels']:
            raise RuntimeError('public-only training source gate failed')
        devices = gpus(8)
        command = distributed('projects.evoseg.hallucination.cache_public_mix',
                              ['--manifest', str(manifest), '--output', str(cache)], devices)
        wait([launch('cache_human_gt', command, devices)], 'ENCODING_ORIGINAL_HUMAN_GT')
        count = len(list((cache / 'records').glob('*.json')))
        if count != source['total_records']:
            raise RuntimeError('public training cache has incomplete coverage')
        update(cache_complete=True, cached_records=count)
        released = evaluate('released_holdout', limit=96)
        latest, scores = {}, {}
        for steps in (20, 100, 300, 600, 1200, 2000):
            if time.time() >= train_stop:
                break
            devices = gpus(8)
            children = []
            for index, variant in enumerate(('full_view', 'interaction')):
                destination = output / 'training' / variant
                arguments = ['--variant', variant, '--cache', str(cache), '--output', str(destination),
                             '--steps', str(steps), '--stop-at', str(train_stop), '--save-every', '100']
                if variant in latest:
                    arguments += ['--resume', latest[variant]]
                allocated = devices[index * 4:(index + 1) * 4]
                children.append(launch(f'train_{variant}_{steps}',
                                       distributed('projects.evoseg.hallucination.train_fresh', arguments, allocated), allocated))
            update(training_started=True)
            wait(children, 'MATCHED_COUPLED_TRAINING')
            for variant in ('full_view', 'interaction'):
                last = json.loads((output / 'training' / variant / 'LATEST.json').read_text())
                latest[variant] = last['checkpoint']
                scores[variant] = evaluate(f'{variant}_heldout_{steps}', latest[variant], limit=96)
            qualified = {}
            for variant, score in scores.items():
                qualified[variant] = (score['positive_gIoU'] >= released['positive_gIoU'] - .01 and
                                      score['T_acc'] >= released['T_acc'] - .01 and
                                      score['N_acc'] >= released['N_acc'] - .01)
            update(last_steps=steps, latest_checkpoints=latest, diagnostic_scores=scores, qualified=qualified)
            if not all(qualified.values()):
                update(status='DIAGNOSTIC_GUARD_FAILED', larger_training_withheld=True)
                break
        if not latest:
            raise RuntimeError('no training checkpoint before reserved evaluation window')
        # Full public validation is distinct from the private train holdout.
        final = {}
        for benchmark in ('gref_val', 'halluseg'):
            for variant in ('released', 'full_view', 'interaction'):
                if time.time() >= deadline - 600:
                    update(full_public_evaluation_remaining=True)
                    break
                final[f'{variant}_{benchmark}'] = evaluate(f'{variant}_{benchmark}',
                    None if variant == 'released' else latest[variant], benchmark=benchmark)
        report = {'status': 'OVERNIGHT_FIRST_RUN_FINISHED', 'deadline': args.deadline,
                  'checkpoints': latest, 'diagnostic_scores': scores, 'public_results': final,
                  'full_public_tables_complete': len(final) == 6, 'sota_claimed': False,
                  'old_faithful_inherited': False, 'generated_training_data': False,
                  'success_means_execution_not_method_superiority': True}
        atomic_json(output / 'FIRST_RUN_REPORT.json', report)
        update(status=report['status'], success=True, child_pids=[], report=str(output / 'FIRST_RUN_REPORT.json'))
        # Keep an honest CPU heartbeat until the requested monitoring time.
        # No GPU holder and no additional training is launched just to occupy cards.
        while time.time() < deadline:
            update(status='RESULTS_READY_MONITORING_UNTIL_DEADLINE')
            time.sleep(min(30, max(1, deadline - time.time())))
        update(status='USER_DEADLINE_REACHED', finished_at_unix=time.time())
    except Exception as error:
        for child in jobs:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        update(status='FAILED_NEEDS_ATTENTION', error=str(error), child_pids=[])
        atomic_json(output / 'FAILURE_REPORT.json', {**state, 'checkpoints_preserved': True})
        raise
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline', default='2026-10-07T14:00:00+08:00')
    p.add_argument('--python', default='/9950backfile/chenjiahui/evo_artifacts/envs/qwen_process_seg/bin/python')
    p.add_argument('--output', default='/9950backfile/chenjiahui/evo_artifacts/results/evoseg_hallucination_20261007/new_public_run')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'LAUNCH.json').exists():
            raise RuntimeError('nightly chain already launched; inspect rather than duplicating')
        command = [args.python, '-m', 'projects.evoseg.hallucination.nightly',
                   '--output', args.output, '--deadline', args.deadline, '--python', args.python]
        with (output / 'DRIVER.log').open('a') as log:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': child.pid, 'deadline': args.deadline, 'command': command})
        print(json.dumps({'supervisor_pid': child.pid, 'deadline': args.deadline}), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()

"""User-authorized longer matched training, with a fixed train-only selection rule."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from projects.evoseg.restart.overnight import available_gpu_indices
from projects.evoseg.restart.prepare_foundation import atomic_json


def milestone_name(variant, step, fork_objective=False):
    return f'{variant}_{"updated_" if fork_objective else ""}all_holdout_{step}'


def check_relaunch(output, resume):
    marker = output / 'LAUNCH.json'
    if not marker.exists():
        return
    if not resume:
        raise RuntimeError('queue already launched; inspect or explicitly resume supervision')
    previous = json.loads(marker.read_text())
    try:
        os.kill(previous['pid'], 0)
    except ProcessLookupError:
        atomic_json(output / f'LAUNCH_previous_{previous["pid"]}.json', previous)
    else:
        raise RuntimeError('previous supervisor is alive; cannot launch a duplicate')


def preservation_guard(reference, candidate, image_ids, seed=42):
    """Paired image-cluster bootstrap, not a public-validation selection rule.

    Predeclared: stop for >3pp mean positive IoU loss, statistically clear
    >1pp loss (upper 95% CI below -1pp), or >2pp presence/absence loss.
    """
    if set(reference) != set(candidate) or set(reference) != set(image_ids):
        raise ValueError('paired diagnostic coverage mismatch')
    groups = {}
    for key in sorted(reference):
        a, b = reference[key], candidate[key]
        if a['empty_gt'] != b['empty_gt']:
            raise ValueError('diagnostic target changed')
        if not a['empty_gt']:
            groups.setdefault(image_ids[key], []).append(b['iou'] - a['iou'])
    if len(groups) < 2:
        raise ValueError('too few positive held-out images')
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(groups), size=(2000, len(groups)))
    means = sums[indices].sum(1) / counts[indices].sum(1)
    mean = float(sums.sum() / counts.sum())
    lower, upper = map(float, np.quantile(means, [.025, .975]))

    def accuracy(values, empty):
        selected = [v for v in values.values() if v['empty_gt'] == empty]
        if not selected:
            raise ValueError('diagnostic requires positive and absent examples')
        return sum(v['predicted_empty'] == empty for v in selected) / len(selected)

    n_delta = accuracy(candidate, True) - accuracy(reference, True)
    t_delta = accuracy(candidate, False) - accuracy(reference, False)
    qualified = mean >= -.03 and upper >= -.01 and n_delta >= -.02 and t_delta >= -.02
    return {'qualified': bool(qualified), 'positive_iou_delta': mean,
            'positive_iou_delta_ci95': [lower, upper], 'N_acc_delta': n_delta,
            'T_acc_delta': t_delta, 'positive_image_clusters': len(groups),
            'selection_uses_public_benchmark': False}


def execute(args):
    root = Path(__file__).resolve().parents[3]
    source, output = Path(args.source_run), Path(args.output)
    cache = Path(args.train_cache) if args.train_cache else source / 'cache'
    output.mkdir(parents=True, exist_ok=True)
    recovered = json.loads((output / 'CONTINUE_STATUS.json').read_text()) if args.recover_completed_training else {}
    lock = (output / 'CONTINUE.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    train_stop = deadline - args.reserve_minutes * 60
    env = dict(os.environ, PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', MASK_TOKENIZER_NUM_MASK_TOKEN='1')
    state = {'pid': os.getpid(), 'status': 'PREPARING_CONTINUATION', 'deadline': args.deadline,
             'source_run': str(source), 'training_stop_unix': train_stop, 'success': False,
             'milestones': args.milestones, 'training_data_changed': bool(args.train_cache),
             'prompt_rollout_rate': args.prompt_rollout_rate,
             'public_validation_selection': False, 'sota_claimed': False}
    children, logs = [], []

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'CONTINUE_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def devices():
        while time.time() < train_stop:
            available = available_gpu_indices(8)
            if available:
                return available
            update(status='WAITING_FOR_FREE_GPUS')
            time.sleep(20)
        raise RuntimeError('no free eight-GPU training window; no foreign jobs touched')

    def launch(name, command, allocated=None):
        child_env = dict(env)
        if allocated is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, allocated))
        log = (output / (name + '.log')).open('a')
        logs.append(log)
        child = subprocess.Popen(command, cwd=root, env=child_env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        update(stage=name, child_pids=[p.pid for p in children if p.poll() is None])
        return child

    def wait(jobs, status, progress_roots=()):
        signature, active = None, time.monotonic()
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None, 0) for p in jobs):
                raise RuntimeError(status + ' worker failed; inspect logs')
            current = tuple(sorted((str(p), p.stat().st_mtime_ns)
                            for directory in progress_roots for p in directory.glob('STATUS_rank*.json')))
            if current != signature:
                signature, active = current, time.monotonic()
            if time.monotonic() - active > 1200:
                raise RuntimeError(status + ' stalled for 20 minutes')
            if time.time() >= deadline:
                raise RuntimeError('user deadline reached; owned workers only will be terminated')
            update(status=status, child_pids=[p.pid for p in jobs if p.poll() is None])
            time.sleep(15)
        if any(p.returncode for p in jobs):
            raise RuntimeError(status + ' failed')

    def distributed(module, arguments, allocated):
        return [args.python, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={len(allocated)}', '-m', module, *arguments]

    def evaluate(name, checkpoint, allocated, benchmark='holdout'):
        directory = output / 'evaluation' / name
        arguments = ['--cache', str(cache), '--output', str(directory), '--benchmark', benchmark]
        if checkpoint:
            arguments += ['--checkpoint', checkpoint]
        wait([launch('eval_' + name, distributed('projects.evoseg.hallucination.eval_fresh', arguments, allocated), allocated)],
             'EVALUATING_' + name, [directory])
        wait([launch('summary_' + name, [args.python, '-m', 'projects.evoseg.hallucination.eval_fresh',
                                       *arguments, '--summarize'])], 'SUMMARIZING_' + name)
        return json.loads((directory / 'METRICS.json').read_text())

    def case_metrics(name):
        return {p.stem: json.loads(p.read_text())['metrics']
                for p in (output / 'evaluation' / name / 'cases').glob('*.json')}

    try:
        previous = json.loads((source / 'FIRST_RUN_REPORT.json').read_text())
        if previous['old_faithful_inherited'] or previous['generated_training_data']:
            raise RuntimeError('requires the public-only fresh-model parent')
        latest = dict(previous['checkpoints'])
        completed_step = 0
        if args.recover_completed_training:
            saved = {v: json.loads((output / 'training' / v / 'LATEST.json').read_text())
                     for v in ('full_view', 'interaction')}
            if len({value['step'] for value in saved.values()}) != 1:
                raise RuntimeError('recovery requires both matched arms saved at the same step')
            completed_step = saved['interaction']['step']
            latest = {v: value['checkpoint'] for v, value in saved.items()}
        from .train_fresh import cached_records
        image_ids = {r['id']: r['image_id'] for r in cached_records(cache) if r['holdout']}
        allocated = devices()
        released = evaluate('released_all_holdout', None, allocated)
        reference = case_metrics('released_all_holdout')
        if len(reference) != len(image_ids):
            raise RuntimeError('incomplete expanded private holdout')
        update(holdout_cases=len(image_ids), reference=released)
        # Preserve 100-step parents as candidates. No score from public val/test
        # is consulted by this script, including its checkpoint selection.
        best = dict(recovered.get('best_checkpoints', {}))
        best_scores = {}
        history = list(recovered.get('history', []))
        for item in history:
            for variant in ('full_view', 'interaction'):
                if item['guards'][variant]['qualified'] and (variant not in best_scores or
                        item['scores'][variant]['gIoU'] > best_scores[variant]['gIoU']):
                    best[variant], best_scores[variant] = item['checkpoints'][variant], item['scores'][variant]
        for variant in ('full_view', 'interaction'):
            name = variant + '_all_holdout_100'
            score = evaluate(name, previous['checkpoints'][variant], allocated)
            guard = preservation_guard(reference, case_metrics(name), image_ids)
            if guard['qualified'] and not args.fork_objective:
                best[variant], best_scores[variant] = latest[variant], score
            update(initial_guard={**state.get('initial_guard', {}), variant: guard})
        for step in args.milestones:
            if step < completed_step:
                continue
            if time.time() >= train_stop:
                break
            allocated = devices()
            jobs, progress = [], []
            for index, variant in enumerate(('full_view', 'interaction')) if step > completed_step else ():
                destination = output / 'training' / variant
                arguments = ['--variant', variant, '--cache', str(cache), '--output', str(destination),
                             '--steps', str(step), '--stop-at', str(train_stop), '--save-every', '100',
                             '--prompt-rollout-rate', str(args.prompt_rollout_rate),
                             '--init-from' if args.fork_objective and not history and not completed_step else '--resume', latest[variant]]
                group = allocated[index * 4:(index + 1) * 4]
                jobs.append(launch(f'train_{variant}_{step}',
                                   distributed('projects.evoseg.hallucination.train_fresh', arguments, group), group))
                progress.append(destination)
            wait(jobs, 'MATCHED_LONGER_TRAINING', progress)
            guards, scores = {}, {}
            for variant in ('full_view', 'interaction'):
                saved = json.loads((output / 'training' / variant / 'LATEST.json').read_text())
                latest[variant] = saved['checkpoint']
                # Parent 100 and extra/fork update 100 are DIFFERENT weights.
                # Never reuse the parent's prediction directory for this stage.
                name = milestone_name(variant, saved['step'], args.fork_objective)
                scores[variant] = evaluate(name, latest[variant], allocated)
                guards[variant] = preservation_guard(reference, case_metrics(name), image_ids)
                if guards[variant]['qualified'] and (variant not in best_scores or
                        scores[variant]['gIoU'] > best_scores[variant]['gIoU']):
                    best[variant], best_scores[variant] = latest[variant], scores[variant]
            history.append({'requested_step': step, 'scores': scores, 'guards': guards, 'checkpoints': dict(latest)})
            completed_step = step
            update(latest_checkpoints=latest, best_checkpoints=best, history=history)
            if not all(v['qualified'] for v in guards.values()):
                update(status='EXPANDED_HOLDOUT_GUARD_STOPPED_LONGER_TRAINING')
                break
        public = {}
        # Previously observed full-Hallu inference takes about 40 minutes on 8
        # cards; only launch if that budget plus cleanup is genuinely available.
        if not args.skip_public and 'interaction' in best and deadline - time.time() > 45 * 60:
            public = evaluate('selected_interaction_halluseg', best['interaction'], allocated, 'halluseg')
        report = {**state, 'status': 'CONTINUATION_FINISHED', 'success': True,
                  'latest_checkpoints': latest, 'best_checkpoints': best,
                  'best_private_scores': best_scores, 'history': history,
                  'new_public_halluseg': public, 'public_gref_retest_pending': True,
                  'full_public_comparison_pending': True, 'selection_uses_public_benchmark': False}
        atomic_json(output / 'CONTINUE_REPORT.json', report)
        update(status='CONTINUATION_FINISHED', success=True, child_pids=[], report=str(output / 'CONTINUE_REPORT.json'))
    except Exception as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        update(status='FAILED_CHECKPOINTS_PRESERVED', error=str(error))
        raise
    finally:
        for log in logs:
            log.close()


def main():
    base = '/9950backfile/chenjiahui/evo_artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=base + '/results/evoseg_hallucination_20261007/new_public_run')
    p.add_argument('--output', default=base + '/results/evoseg_hallucination_20261007/longer_public_run')
    p.add_argument('--python', default=base + '/envs/qwen_process_seg/bin/python')
    p.add_argument('--deadline', default='2026-10-07T14:00:00+08:00')
    p.add_argument('--reserve-minutes', type=int, default=60)
    p.add_argument('--milestones', nargs='+', type=int, default=[300, 600, 1200])
    p.add_argument('--train-cache', help='separately audited public cache; requires --fork-objective')
    p.add_argument('--prompt-rollout-rate', type=float, default=0.)
    p.add_argument('--fork-objective', action='store_true', help='initialize weights; reset optimizer for changed objective/data')
    p.add_argument('--skip-public', action='store_true', help='leave full public retest to the shared queued evaluation stage')
    p.add_argument('--resume-supervision', action='store_true', help='relaunch only after previous supervisor has exited')
    p.add_argument('--recover-completed-training', action='store_true', help='reuse saved matched weights and evaluate the interrupted milestone')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.milestones != sorted(set(args.milestones)) or min(args.milestones) < (1 if args.fork_objective else 101):
        p.error('milestones must be unique increasing positive steps (above 100 for unchanged continuation)')
    if (args.train_cache or args.prompt_rollout_rate) and not args.fork_objective:
        p.error('a changed objective/data requires --fork-objective, not a silent optimizer resume')
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'LAUNCH.json').exists():
            if not args.resume_supervision:
                raise RuntimeError('continuation already launched; inspect instead of duplicating')
            previous = json.loads((output / 'LAUNCH.json').read_text())
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('previous supervisor is still alive; cannot duplicate it')
            atomic_json(output / f'LAUNCH_previous_{previous["pid"]}.json', previous)
        command = [args.python, '-m', 'projects.evoseg.hallucination.continue_training',
                   '--source-run', args.source_run, '--output', args.output, '--deadline', args.deadline,
                   '--reserve-minutes', str(args.reserve_minutes), '--milestones', *map(str, args.milestones)]
        for option in ('fork_objective', 'skip_public', 'recover_completed_training'):
            if getattr(args, option):
                command += ['--' + option.replace('_', '-')]
        if args.train_cache:
            command += ['--train-cache', args.train_cache]
        command += ['--prompt-rollout-rate', str(args.prompt_rollout_rate)]
        with (output / 'DRIVER.log').open('a') as log:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': child.pid, 'command': command, 'deadline': args.deadline})
        starting = json.loads((output / 'CONTINUE_STATUS.json').read_text()) if (output / 'CONTINUE_STATUS.json').exists() else {}
        atomic_json(output / 'CONTINUE_STATUS.json', {**starting, 'pid': child.pid,
                    'status': 'SUPERVISOR_STARTING', 'deadline': args.deadline, 'updated_at_unix': time.time()})
        print(json.dumps({'supervisor_pid': child.pid, 'deadline': args.deadline}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

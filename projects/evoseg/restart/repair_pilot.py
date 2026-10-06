"""Controlled ungated-head repair, public sampling comparison, guarded continuation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .native_data import split_video_ids
from .overnight import available_gpu_indices
from .prepare_foundation import atomic_json


def heldout_keys(videos):
    _, held = split_video_ids(videos)
    keys = []
    for video in held:
        expressions = videos[video]['expressions']
        rng = random.Random('repair-heldout-42:' + video)
        positive = [key for key in sorted(expressions) if expressions[key]['anno_id']]
        negative = [key for key in sorted(expressions) if not expressions[key]['anno_id']]
        keys.extend([[video, key] for key in rng.sample(positive, min(2, len(positive)))])
        keys.extend([[video, key] for key in rng.sample(negative, min(1, len(negative)))])
    return keys


def presence_summary(meta_path, mask_path, results_path):
    from pycocotools import mask as mask_utils
    metadata = json.loads(Path(meta_path).read_text())['videos']
    annotations = json.loads(Path(mask_path).read_text())
    predictions = json.loads(Path(results_path).read_text())
    present, missing, absent, false_positive = 0, 0, 0, 0
    for video, expressions in predictions.items():
        for key, item in expressions.items():
            anno_ids = metadata[video]['expressions'][key]['anno_id']
            for frame, prediction in enumerate(item['prediction_masks']):
                exists = any(annotations[str(a)][frame] is not None and
                             int(mask_utils.area(annotations[str(a)][frame])) > 0 for a in anno_ids)
                predicted = int(mask_utils.area(prediction)) > 0
                present += int(exists)
                missing += int(exists and not predicted)
                absent += int(not exists)
                false_positive += int(not exists and predicted)
    return {'present_frames': present, 'empty_on_present': missing,
            'empty_on_present_fraction': missing / present if present else None,
            'absent_frames': absent, 'nonempty_on_absent': false_positive,
            'nonempty_on_absent_fraction': false_positive / absent if absent else None}


def passes_guard(baseline, candidate, max_drop=.005, max_empty_increase=.01):
    before, after = baseline['metrics'], candidate['metrics']
    positive_before, positive_after = before['target_present']['j_and_f'], after['target_present']['j_and_f']
    empty_before = baseline['presence']['empty_on_present_fraction']
    empty_after = candidate['presence']['empty_on_present_fraction']
    if any(value is None for value in (positive_before, positive_after, empty_before, empty_after)):
        return False
    return (after['j_and_f'] >= before['j_and_f'] - max_drop and
            positive_after >= positive_before - max_drop and
            empty_after <= empty_before + max_empty_increase)


class RepairRunner:
    def __init__(self, args):
        self.args = args
        self.root = Path(__file__).resolve().parents[3]
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run_dir / 'RUN.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = {'status': 'PREPARING', 'pid': os.getpid(), 'training_started': False,
                      'started_at_unix': time.time(), 'success': False}
        self.env = dict(os.environ)
        self.env.update(PYTHONPATH=f'{args.overlay}:{self.root}', HF_HUB_OFFLINE='1',
                        HF_MODULES_CACHE=str(self.run_dir / 'hf_modules_cache'), OMP_NUM_THREADS='4',
                        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                        MPLCONFIGDIR=str(self.run_dir / 'matplotlib_cache'))

    def update(self, **fields):
        self.state.update(fields, updated_at_unix=time.time())
        atomic_json(self.run_dir / 'STATUS.json', self.state)
        print(json.dumps(self.state), flush=True)

    def wait_gpus(self):
        while True:
            indices = available_gpu_indices(self.args.gpus)
            if indices:
                self.env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, indices))
                self.update(gpu_indices=indices)
                return
            if time.time() >= self.args.stop_at:
                raise RuntimeError('free GPU wait exceeded the bounded pilot window')
            self.update(status='WAITING_FOR_FREE_GPUS')
            time.sleep(30)

    def child(self, stage, command, train_status=None):
        self.update(status=stage, stage_log=str(self.run_dir / (stage + '.log')))
        with (self.run_dir / (stage + '.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=self.root, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.update(child_pid=child.pid)
            while child.poll() is None:
                time.sleep(15)
                detail = {}
                if train_status and Path(train_status).exists():
                    training = json.loads(Path(train_status).read_text())
                    detail = {'training_started': training['training_started'],
                              'training_step': training.get('step', 0),
                              'seconds_per_update': training.get('seconds_per_update')}
                self.update(child_pid=child.pid, **detail)
            if child.returncode:
                raise RuntimeError(f'{stage} failed with exit {child.returncode}; inspect its stage log')
        self.update(child_pid=None)

    def launch(self, module, arguments):
        return [self.args.python, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={self.args.gpus}', '-m', module, *arguments]

    def evaluate(self, name, split_root, keys=None, checkpoint=None):
        self.wait_gpus()
        output = self.run_dir / name
        meta, masks = Path(split_root) / 'meta_expressions_v2.json', Path(split_root) / 'mask_dict.json'
        arguments = ['--model-dir', self.args.model_dir, '--assets', self.args.assets,
                     '--images', str(Path(split_root) / 'JPEGImages'), '--meta', str(meta), '--output', str(output)]
        if keys:
            arguments.extend(['--keys', str(keys)])
        if checkpoint:
            arguments.extend(['--checkpoint', str(checkpoint)])
        self.child(name, self.launch('projects.evoseg.restart.native_eval', arguments))
        from .native_eval import merge_predictions
        selected = json.loads(Path(keys).read_text()) if keys else None
        count = merge_predictions(meta, output, selected)
        self.child(name + '_metrics', [self.args.python, '-m', 'projects.evoseg.eval.eval_mevis_jf',
            str(output / 'results.json'), '--meta', str(meta), '--mask', str(masks), '--workers', '8',
            '--output', str(output / 'metrics.json')])
        metrics = json.loads((output / 'metrics.json').read_text())
        if metrics['evaluated_pairs'] != count or metrics['j_and_f'] is None:
            raise RuntimeError('incomplete controlled evaluation coverage')
        presence = presence_summary(meta, masks, output / 'results.json')
        atomic_json(output / 'presence.json', presence)
        return {'metrics': metrics, 'presence': presence, 'checkpoint': str(checkpoint) if checkpoint else None}

    def train(self, name, stop_updates, probability=None, resume=None):
        self.wait_gpus()
        output = self.run_dir / name
        arguments = ['--model-dir', self.args.model_dir, '--assets', self.args.assets,
                     '--train-root', self.args.train_root, '--output', str(output),
                     '--max-updates', '200', '--stop-updates', str(stop_updates),
                     '--stop-at', str(self.args.stop_at)]
        if probability is not None:
            arguments.extend(['--negative-query-probability', str(probability)])
        if resume:
            arguments.extend(['--resume', str(resume)])
        self.child(name + f'_train_to_{stop_updates}',
                   self.launch('projects.evoseg.restart.native_train', arguments), output / 'TRAINING_STATUS.json')
        status = json.loads((output / 'TRAINING_STATUS.json').read_text())
        if status['step'] != stop_updates:
            raise RuntimeError('training reached its time limit before this planned diagnostic checkpoint')
        return json.loads((output / 'LATEST.json').read_text())['checkpoint']

    def run(self):
        try:
            assets = json.loads(Path(self.args.assets).read_text())
            if assets['status'] != 'ASSETS_READY':
                raise RuntimeError('repair requires complete released assets')
            self.wait_gpus()
            self.child('real_sam_training_parity', [self.args.python, '-m',
                'projects.evoseg.restart.audit_sam_training', '--model-dir', self.args.model_dir,
                '--output', str(self.run_dir / 'SAM_TRAINING_PARITY.json')])
            videos = json.loads((Path(self.args.train_root) / 'meta_expressions_v2.json').read_text())['videos']
            keys = self.run_dir / 'HELD_OUT_KEYS.json'
            atomic_json(keys, heldout_keys(videos))
            baseline = self.evaluate('baseline_heldout', self.args.train_root, keys)
            candidates = []
            for name, probability in (('ungated_only', None), ('ungated_balanced10', .1)):
                checkpoint = self.train(name, 50, probability)
                score = self.evaluate(name + '_step50_heldout', self.args.train_root, keys, checkpoint)
                candidates.append({'name': name, 'negative_query_probability': probability,
                                   'score': score, 'qualified': passes_guard(baseline, score)})
                atomic_json(self.run_dir / 'PILOT_COMPARISON.json', {'baseline': baseline,
                    'candidates': candidates, 'scope': 'held-out public train diagnostic, not benchmark/SOTA',
                    'max_allowed_jf_drop_points': .5, 'max_empty_on_present_increase_points': 1.})
            qualified = [candidate for candidate in candidates if candidate['qualified']]
            if not qualified:
                self.update(status='PILOTS_FAILED_GUARD', report=str(self.run_dir / 'PILOT_COMPARISON.json'),
                            success=False, full_training_started=False)
                return
            winner = max(qualified, key=lambda c: (c['score']['metrics']['target_present']['j_and_f'],
                                                   c['score']['metrics']['j_and_f']))
            checkpoint = winner['score']['checkpoint']
            self.update(selected_variant=winner['name'], full_training_started=True)
            for step in (100, 200):
                checkpoint = self.train(winner['name'], step, winner['negative_query_probability'], checkpoint)
                score = self.evaluate(winner['name'] + f'_step{step}_heldout', self.args.train_root, keys, checkpoint)
                atomic_json(self.run_dir / f'GATE_step{step}.json', {'score': score, 'baseline': baseline,
                            'passed': passes_guard(baseline, score)})
                if not passes_guard(baseline, score):
                    self.update(status='CONTINUATION_FAILED_GUARD', failed_step=step,
                                report=str(self.run_dir / f'GATE_step{step}.json'), success=False)
                    return
            fine = self.evaluate(winner['name'] + '_mevis_v2_full', self.args.valid_root, checkpoint=checkpoint)
            reference = Path(self.args.baseline_root)
            original = json.loads((reference / 'baseline_mevis_v2/metrics.json').read_text())
            loading = json.loads((reference / 'baseline_mevis_v2/LOADING_rank0.json').read_text())
            if loading['revision'] != assets['revision'] or original['evaluated_pairs'] != 907 or fine['metrics']['evaluated_pairs'] != 907:
                raise RuntimeError('published baseline and repaired full evaluation are not compatible')
            report = {'selected_variant': winner['name'], 'checkpoint': checkpoint,
                      'baseline_metrics_reused_from': str(reference / 'baseline_mevis_v2/metrics.json'),
                      'baseline': original, 'repaired': fine,
                      'delta_jf_points': 100 * (fine['metrics']['j_and_f'] - original['j_and_f']),
                      'pilot_comparison': str(self.run_dir / 'PILOT_COMPARISON.json'), 'sota_claimed': False}
            atomic_json(self.run_dir / 'REPAIR_REPORT.json', report)
            self.update(status='REPAIR_EVALUATED', success=True, report=str(self.run_dir / 'REPAIR_REPORT.json'))
        except Exception as error:
            self.update(status='FAILED_NEEDS_ATTENTION', success=False,
                        error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--run-dir', default=str(base / 'results/evoseg_restart_20261006_repair'))
    parser.add_argument('--baseline-root', default=str(base / 'results/evoseg_restart_20261006'))
    parser.add_argument('--assets', default=str(base / 'results/evoseg_restart_20261006/ASSETS.json'))
    parser.add_argument('--model-dir', default=str(base / 'models/SaSaSa2VA-26B-official'))
    parser.add_argument('--train-root', default=str(base / 'datasets/mevis_v2/train'))
    parser.add_argument('--valid-root', default=str(base / 'datasets/mevis_v2/valid_u'))
    parser.add_argument('--python', default=str(base / 'envs/virst/bin/python'))
    parser.add_argument('--overlay', default=str(base / 'envs/sasasa2va_native_overlay'))
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--stop-at', type=float, default=time.time() + 5 * 3600)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.gpus < 1 or args.stop_at <= time.time():
        parser.error('need positive GPU count and a future bounded end time')
    if args.detach:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / 'LAUNCH.json'
        if marker.exists():
            raise RuntimeError('repair launch already exists; inspect before any relaunch')
        command = [sys.executable, '-m', 'projects.evoseg.restart.repair_pilot']
        for key, value in vars(args).items():
            if key != 'detach':
                command.extend(['--' + key.replace('_', '-'), str(value)])
        with (run_dir / 'driver.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'created_at_unix': time.time()})
        print(json.dumps({'pid': process.pid, 'status': 'LAUNCHED', 'training_started': False}), flush=True)
    else:
        RepairRunner(args).run()


if __name__ == '__main__':
    main()

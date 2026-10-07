"""Bounded CPU-only interface queue; leaves the public GPU supervisor untouched."""
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


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=20)


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'QUEUE.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    state = {'pid': os.getpid(), 'status': 'WAITING_FOR_REAL_IMAGE_SMOKE',
             'deadline': args.deadline, 'device': 'cpu', 'limit': args.limit,
             'gpu_jobs_untouched': True, 'training_performed': False,
             'paper_score_reproduction': False, 'success': False}
    child = None

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'QUEUE_STATUS.json', state)
        print(json.dumps(state), flush=True)

    try:
        dependency = Path(args.after_report)
        while not dependency.exists():
            if time.time() >= deadline:
                raise TimeoutError('deadline reached while waiting for smoke')
            if (dependency.parent / 'ERROR.json').exists():
                raise RuntimeError('real-image interface smoke failed; no larger run started')
            update()
            time.sleep(15)
        smoke = json.loads(dependency.read_text())
        if smoke['candidates_with_points'] < 1:
            raise RuntimeError('smoke never exercised real point prompts; no larger run started')
        if time.time() >= deadline:
            raise TimeoutError('deadline reached before starting diagnostic')
        root = Path(__file__).resolve().parents[3]
        arguments = ['--output', str(output / 'diagnostic'), '--device', 'cpu',
                     '--cpu-threads', '4', '--limit', str(args.limit)]
        if args.cache:
            arguments += ['--cache', args.cache]
        command = [args.python, '-m', 'projects.evoseg.hallucination.run_read_alignment', *arguments]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                   PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
                   MASK_TOKENIZER_NUM_MASK_TOKEN='1', HF_HUB_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        with (output / 'WORKER.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('deadline reached; completed diagnostic cases are preserved')
                status = output / 'diagnostic/STATUS.json'
                progress = json.loads(status.read_text()) if status.exists() else {}
                update(status='CPU_PRIVATE_DIAGNOSTIC', child_pid=child.pid,
                       cases_done=progress.get('cases_done', 0), worker_status=progress.get('status'))
                time.sleep(15)
            if child.returncode:
                raise RuntimeError('CPU interface diagnostic failed; inspect WORKER.log')
        report = json.loads((output / 'diagnostic/REPORT.json').read_text())
        update(status='CPU_PRIVATE_DIAGNOSTIC_COMPLETE', child_pid=None, success=True,
               cases_done=report['cases'], candidates_with_points=report['candidates_with_points'])
    except Exception as error:
        stop_owned(child)
        update(status='DEADLINE_REACHED_PARTIAL_RESULTS' if isinstance(error, TimeoutError)
               else 'FAILED_PREDICTIONS_PRESERVED', child_pid=None, error=str(error))
        if not isinstance(error, TimeoutError):
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--after-report', required=True)
    p.add_argument('--cache')
    p.add_argument('--limit', type=int, default=96)
    p.add_argument('--deadline', required=True)
    p.add_argument('--python', default='/9950backfile/chenjiahui/evo_artifacts/envs/qwen_process_seg/bin/python')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        check_relaunch(output, False)
        command = [args.python, '-m', 'projects.evoseg.hallucination.queue_read_alignment',
                   '--output', args.output, '--after-report', args.after_report,
                   '--limit', str(args.limit), '--deadline', args.deadline]
        if args.cache:
            command += ['--cache', args.cache]
        with (output / 'DRIVER.log').open('a') as log:
            child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': child.pid, 'command': command})
        print(json.dumps({'queue_pid': child.pid, 'deadline': args.deadline, 'device': 'cpu'}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

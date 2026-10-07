"""User-authorized GPU sharing with a bounded monitoring handoff.

Pause ONLY our public supervisor, not its eight evaluation workers. Monitor
GPU conflicts here while an interface diagnostic shares one card. Restore
the public supervisor on success, error, timeout and catchable termination.
No foreign task is ever signalled. CPU/GPU prediction caches stay separate.
"""
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
from .queue_read_alignment import stop_owned


def owns_command(pid, module, output):
    try:
        argv = [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
    except (FileNotFoundError, PermissionError):
        return False
    return any(argv[i:i+2] == ['-m', module] for i in range(len(argv)-1)) and any(
        argv[i:i+2] == ['--output', str(output)] for i in range(len(argv)-1))


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return {int(parts[0]): (parts[1], int(parts[2])) for line in result.stdout.splitlines()
            if (parts := [v.strip() for v in line.split(',')])}


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [(parts[0], int(parts[1]), int(parts[2])) for line in result.stdout.splitlines()
            if (parts := [v.strip() for v in line.split(',')])]


def execute(args):
    output, public = map(Path, (args.output, args.public_output))
    cpu = Path(args.cpu_output) if args.cpu_output else None
    job = getattr(args, 'job', 'read_alignment')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'SHARE.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    public_module = 'projects.evoseg.hallucination.recover_evaluations'
    read_module = ('projects.evoseg.hallucination.evidence_pilot' if job == 'evidence_pilot'
                   else 'projects.evoseg.hallucination.run_read_alignment')
    state = {'pid': os.getpid(), 'status': 'CHECKING_OWNERSHIP', 'gpu': args.gpu,
             'public_supervisor_pid': args.public_supervisor, 'deadline': args.deadline,
             'public_workers_paused': False, 'training_performed': False, 'job': job, 'success': False}
    paused, child = False, None

    def update(**values):
        state.update(values, updated_at_unix=time.time())
        atomic_json(output / 'SHARE_STATUS.json', state)
        print(json.dumps(state), flush=True)

    def terminate(signum, frame):
        raise RuntimeError('GPU sharing interrupted by signal ' + str(signum))

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        update()
        if datetime.fromisoformat(args.deadline).timestamp() <= time.time() or args.max_seconds <= 0:
            raise RuntimeError('sharing deadline already reached; no process signalled')
        if not owns_command(args.public_supervisor, public_module, public):
            raise RuntimeError('public supervisor identity mismatch; no process signalled')
        if bool(args.cpu_supervisor) != bool(cpu):
            raise RuntimeError('CPU supervisor/output must be supplied together')
        if cpu and not owns_command(args.cpu_supervisor, 'projects.evoseg.hallucination.queue_read_alignment', cpu):
            raise RuntimeError('CPU supervisor identity mismatch; no process signalled')
        old = json.loads((public / 'RECOVERY_STATUS.json').read_text())
        destination = Path(old['prediction_directory'])
        inventory = gpu_inventory()
        if args.gpu not in inventory or inventory[args.gpu][1] < 48000:
            raise RuntimeError('less than 48 GiB spare: do not start a second model')
        selected_uuids = {inventory[index][0] for index in old['devices']}
        processes = gpu_processes()
        public_pids = {pid for uuid, pid, memory in processes if uuid in selected_uuids and
                       owns_command(pid, 'projects.evoseg.hallucination.eval_fresh', destination)}
        if any(uuid in selected_uuids and memory >= 1024 and pid not in public_pids
               for uuid, pid, memory in processes):
            raise RuntimeError('foreign GPU task present; no sharing started')
        if cpu:
            cpu_status = json.loads((cpu / 'QUEUE_STATUS.json').read_text())
            cpu_child = cpu_status.get('child_pid')
            if cpu_child and not owns_command(cpu_child, 'projects.evoseg.hallucination.run_read_alignment', cpu / 'diagnostic'):
                raise RuntimeError('CPU worker identity mismatch; no process signalled')
            # Exact, verified task PIDs only. Preserve every completed CPU case.
            os.kill(args.cpu_supervisor, signal.SIGTERM)
            if cpu_child:
                os.kill(cpu_child, signal.SIGTERM)
            atomic_json(cpu / 'QUEUE_STATUS.json', {**cpu_status, 'status': 'STOPPED_FOR_GPU_SWITCH',
                        'child_pid': None, 'success': False, 'predictions_preserved': True,
                        'updated_at_unix': time.time()})
        # This transfers monitoring; it does NOT pause public GPU computation.
        paused = True
        os.kill(args.public_supervisor, signal.SIGSTOP)
        update(status='MONITORING_HANDOFF', public_supervisor_paused=True,
               public_worker_pids=sorted(public_pids), cpu_stopped=bool(cpu))
        root = Path(__file__).resolve().parents[3]
        diagnostic = output / 'diagnostic'
        command = [args.python, '-m', read_module, '--output', str(diagnostic), '--gpu-memory-fraction', '.4']
        if job == 'evidence_pilot':
            command += ['--train-limit', str(args.train_limit), '--holdout-limit', str(args.limit),
                        '--steps', str(args.steps)]
            if args.reuse_inputs:
                command += ['--reuse-inputs', args.reuse_inputs]
        else:
            command += ['--device', 'cuda', '--limit', str(args.limit)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                   PYTHONPATH=str(root) + ':/9950backfile/chenjiahui/evo_artifacts/envs/hallucination_public_overlay',
                   MASK_TOKENIZER_NUM_MASK_TOKEN='1', HF_HUB_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        deadline = min(datetime.fromisoformat(args.deadline).timestamp(), time.time() + args.max_seconds)
        with (output / 'GPU_WORKER.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('bounded shared-GPU diagnostic deadline reached')
                foreign = []
                for uuid, pid, memory in gpu_processes():
                    if uuid not in selected_uuids or memory < 1024:
                        continue
                    if pid in public_pids and owns_command(pid, 'projects.evoseg.hallucination.eval_fresh', destination):
                        continue
                    if pid == child.pid and owns_command(pid, read_module, diagnostic):
                        continue
                    foreign.append(pid)
                if foreign:
                    raise RuntimeError('foreign GPU task detected; yield private test: ' + str(foreign))
                if gpu_inventory()[args.gpu][1] < 8192:
                    raise RuntimeError('less than 8 GiB shared-card headroom; yield private test')
                worker_path = diagnostic / 'STATUS.json'
                worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
                update(status='GPU_PRIVATE_DIAGNOSTIC', child_pid=child.pid,
                       cases_done=worker.get('cases_done', 0), expected_cases=args.limit,
                       peak_cuda_allocated_mib=worker.get('peak_cuda_allocated_mib'),
                       worker_status=worker.get('status'), training_step=worker.get('step'),
                       input_features_done=worker.get('input_features_done'))
                time.sleep(5)
            if child.returncode:
                raise RuntimeError('GPU interface diagnostic failed; inspect GPU_WORKER.log')
        report = json.loads((diagnostic / 'REPORT.json').read_text())
        update(status='GPU_PRIVATE_DIAGNOSTIC_COMPLETE', success=True,
               cases_done=report.get('cases', report.get('reference', {}).get('cases')), child_pid=None,
               training_performed=report.get('training_performed', False))
    except Exception as error:
        stop_owned(child)
        update(status='PARTIAL_PREDICTIONS_PRESERVED', error=str(error), child_pid=None)
    finally:
        if paused and owns_command(args.public_supervisor, public_module, public):
            os.kill(args.public_supervisor, signal.SIGCONT)
            update(public_supervisor_paused=False, public_supervisor_restored=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--public-output', required=True)
    p.add_argument('--cpu-output')
    p.add_argument('--public-supervisor', type=int, required=True)
    p.add_argument('--cpu-supervisor', type=int)
    p.add_argument('--job', choices=['read_alignment', 'evidence_pilot'], default='read_alignment')
    p.add_argument('--train-limit', type=int, default=128)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--reuse-inputs')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--limit', type=int, default=96)
    p.add_argument('--deadline', required=True)
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--python', default='/9950backfile/chenjiahui/evo_artifacts/envs/qwen_process_seg/bin/python')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        check_relaunch(output, False)
        command = [args.python, '-m', 'projects.evoseg.hallucination.share_read_gpu']
        for name in ('output', 'public_output', 'cpu_output', 'public_supervisor', 'cpu_supervisor',
                     'gpu', 'limit', 'deadline', 'max_seconds', 'job', 'train_limit', 'steps', 'reuse_inputs'):
            if getattr(args, name) is not None:
                command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
        with (output / 'DRIVER.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / 'LAUNCH.json', {'pid': process.pid, 'command': command})
        print(json.dumps({'gpu_share_pid': process.pid, 'gpu': args.gpu}))
    else:
        execute(args)


if __name__ == '__main__':
    main()

"""Fetch one pinned public foundation; never start training on incomplete weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def validate_inventory(model_dir: Path, weights: list[dict]) -> None:
    if not weights:
        raise RuntimeError('empty public weight inventory')
    for item in weights:
        path = model_dir / item['name']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError(f"incomplete public shard: {item['name']}")
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != {item['name'] for item in weights}:
        raise RuntimeError('published shard index and downloaded inventory differ')


def run(args):
    # These settings affect this public download process only. Credentials are
    # deliberately not sent to the user proxy or the public model endpoint.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    from huggingface_hub import HfApi, configure_http_backend, snapshot_download
    import requests

    def backend_factory():
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': args.proxy, 'https': args.proxy}
        return session

    configure_http_backend(backend_factory=backend_factory)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / 'ASSETS.json'
    state = {'status': 'RESOLVING_PUBLIC_REVISION', 'repo_id': args.repo_id,
             'model_directory': str(args.model_dir), 'pid': os.getpid(),
             'native_pixel_model': 'released SAM2; no replacement',
             'training_started': False, 'started_at_unix': time.time()}
    atomic_json(status_path, state)
    finished = threading.Event()
    lock = threading.Lock()

    def progress():
        # One bounded directory, including Hugging Face's resumable partials.
        paths = list(args.model_dir.glob('*.safetensors'))
        paths.extend(args.model_dir.glob('.cache/huggingface/download/*.incomplete'))
        downloaded = 0
        for path in paths:
            try:
                downloaded += path.stat().st_size
            except FileNotFoundError:
                pass
        with lock:
            state.update({'downloaded_weight_bytes_including_partials': downloaded,
                          'updated_at_unix': time.time()})
            atomic_json(status_path, state)

    def heartbeat():
        while not finished.wait(15):
            progress()

    try:
        info = HfApi(token=False).model_info(args.repo_id, revision=args.revision, files_metadata=True, timeout=30)
        weights = [item for item in info.siblings if item.rfilename.endswith('.safetensors')]
        if not weights or any(not item.size for item in weights):
            raise RuntimeError('public weight inventory is empty or lacks file sizes')
        state.update({'status': 'DOWNLOADING', 'revision': info.sha,
                      'expected_weight_bytes': sum(item.size for item in weights),
                      'weight_files': [{'name': item.rfilename, 'size': item.size} for item in weights]})
        atomic_json(status_path, state)
        print(json.dumps({'revision': info.sha, 'expected_weight_bytes': state['expected_weight_bytes']}), flush=True)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        snapshot_download(args.repo_id, revision=info.sha, token=False,
                          local_dir=str(args.model_dir), max_workers=args.workers,
                          allow_patterns=['*.safetensors', '*.json', '*.py', '*.model', '*.txt', '*.md'])
        validate_inventory(args.model_dir, state['weight_files'])
        finished.set()
        thread.join()
        state.update({'status': 'ASSETS_READY', 'finished_at_unix': time.time(),
                      'checks': ['pinned upstream revision', 'all published shard sizes', 'complete shard index'],
                      'next_gate': 'strict complete model load and released native inference; not yet benchmarked'})
        progress()
        print('ASSETS_READY; training has not started.', flush=True)
    except Exception as error:
        finished.set()
        with lock:
            state.update({'status': 'DOWNLOAD_FAILED', 'error_type': type(error).__name__,
                          'error': str(error), 'updated_at_unix': time.time()})
            atomic_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--repo-id', default='QuanzhuNiu/SaSaSa2VA-26B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--model-dir', type=Path, default=root / 'models/SaSaSa2VA-26B-official')
    parser.add_argument('--run-dir', type=Path, default=root / 'results/evoseg_restart_20261006')
    parser.add_argument('--proxy', default='http://127.0.0.1:17890')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.detach:
        marker = args.run_dir / 'DOWNLOAD_LAUNCH.json'
        if marker.exists():
            raise RuntimeError('existing launch detected; inspect it before starting another download')
        command = [sys.executable, '-m', 'projects.evoseg.restart.prepare_foundation',
                   '--repo-id', args.repo_id, '--revision', args.revision,
                   '--model-dir', str(args.model_dir), '--run-dir', str(args.run_dir),
                   '--proxy', args.proxy, '--workers', str(args.workers)]
        root = Path(__file__).resolve().parents[3]
        with (args.run_dir / 'download.log').open('a') as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        payload = {'pid': process.pid, 'command': command, 'working_directory': str(root),
                   'created_at_unix': time.time(), 'training_started': False}
        atomic_json(marker, payload)
        print(json.dumps(payload, indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()

"""Pinned public multi-turn data + released SAMTok; no silent training launch."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from projects.evoseg.restart.prepare_foundation import atomic_json


def run(args):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    from huggingface_hub import HfApi, configure_http_backend, snapshot_download
    import requests

    def backend():
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': args.proxy, 'https': args.proxy}
        return session

    configure_http_backend(backend_factory=backend)
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'PREPARE.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'RESOLVING_PUBLIC_ASSETS', 'pid': os.getpid(),
             'started_at_unix': time.time(), 'training_started': False,
             'benchmark_complete': False, 'native_mask_decoder': 'released SAMTok / SAM2.1'}
    status = root / 'ASSETS.json'
    atomic_json(status, state)
    finished = threading.Event()
    guard = threading.Lock()

    def update(**values):
        with guard:
            state.update(values, updated_at_unix=time.time())
            atomic_json(status, state)

    def heartbeat():
        while not finished.wait(15):
            model = Path(args.model_dir)
            files = list(model.glob('*.safetensors')) + list(model.glob('*.pth')) + list(model.glob('*.pt'))
            files += list(model.glob('.cache/huggingface/download/*.incomplete'))
            sizes = []
            for path in files:
                try:
                    sizes.append(path.stat().st_size)
                except FileNotFoundError:
                    pass
            update(downloaded_model_bytes_including_partials=sum(sizes))

    thread = threading.Thread(target=heartbeat, daemon=True)
    try:
        api = HfApi(token=False)
        data_info = api.repo_info(args.dataset_repo, repo_type='dataset', files_metadata=True)
        selected = [file for file in data_info.siblings if file.rfilename.startswith('conversations_folder/')
                    and Path(file.rfilename).name.startswith(('mr_refcoco', 'mr_paco'))]
        if not selected:
            raise RuntimeError('no published multi-round conversation files')
        model_info = api.repo_info(args.model_repo, repo_type='model', files_metadata=True)
        model_files = [file for file in model_info.siblings if file.rfilename.endswith(('.safetensors', '.pth', '.pt'))]
        if not model_files or any(file.size is None for file in model_files + selected):
            raise RuntimeError('published inventories are incomplete')
        update(dataset_repo=args.dataset_repo, dataset_revision=data_info.sha,
               model_repo=args.model_repo, model_revision=model_info.sha,
               expected_model_bytes=sum(file.size for file in model_files),
               dataset_inventory=[{'name': file.rfilename, 'size': file.size} for file in selected],
               model_inventory=[{'name': file.rfilename, 'size': file.size} for file in model_files],
               status='DOWNLOADING_PUBLIC_DIALOGUES')
        thread.start()
        snapshot_download(args.dataset_repo, repo_type='dataset', revision=data_info.sha,
                          token=False, local_dir=args.dataset_dir, max_workers=4,
                          allow_patterns=[file.rfilename for file in selected])
        for file in selected:
            if (Path(args.dataset_dir) / file.rfilename).stat().st_size != file.size:
                raise RuntimeError('incomplete conversation file: ' + file.rfilename)
        update(status='DOWNLOADING_RELEASED_MODEL', public_dialogues_downloaded=True)
        snapshot_download(args.model_repo, revision=model_info.sha, token=False,
                          local_dir=args.model_dir, max_workers=4,
                          allow_patterns=['*.safetensors', '*.pth', '*.pt', '*.json', '*.txt', '*.jinja', '*.md'])
        for file in model_files:
            if (Path(args.model_dir) / file.rfilename).stat().st_size != file.size:
                raise RuntimeError('incomplete model file: ' + file.rfilename)
        index = json.loads((Path(args.model_dir) / 'model.safetensors.index.json').read_text())
        shards = {file.rfilename for file in model_files if file.rfilename.endswith('.safetensors')}
        if set(index['weight_map'].values()) != shards:
            raise RuntimeError('model shard inventory and index differ')
        finished.set()
        thread.join()
        update(status='ASSETS_DOWNLOADED', finished_at_unix=time.time(),
               downloaded_model_bytes_including_partials=sum(file.size for file in model_files),
               next_gate='audit dialogue schema, mask/image coverage and full native model loading; no training yet')
        print(json.dumps(state), flush=True)
    except Exception as error:
        finished.set()
        update(status='DOWNLOAD_FAILED', error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--run-dir', default=str(base / 'results/evoseg_dialog_20261006'))
    parser.add_argument('--dataset-dir', default=str(base / 'datasets/SegLLM-official'))
    parser.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    parser.add_argument('--dataset-repo', default='Marlo-Z/SegLLM_dataset')
    parser.add_argument('--model-repo', default='zhouyik/Qwen3-VL-8B-SAMTok')
    parser.add_argument('--proxy', default='http://127.0.0.1:17890')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach:
        root = Path(args.run_dir)
        root.mkdir(parents=True, exist_ok=True)
        marker = root / 'PREPARE_LAUNCH.json'
        if marker.exists():
            raise RuntimeError('asset launch already exists; inspect before relaunching')
        command = [sys.executable, '-m', 'projects.evoseg.dialog.prepare_assets']
        for key, value in vars(args).items():
            if key != 'detach':
                command.extend(['--' + key.replace('_', '-'), str(value)])
        with (root / 'download.log').open('a') as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        atomic_json(marker, {'pid': process.pid, 'command': command, 'created_at_unix': time.time()})
        print(json.dumps({'pid': process.pid, 'training_started': False}), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()

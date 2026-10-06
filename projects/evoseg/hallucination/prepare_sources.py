"""Verify bytes against the author's pinned public mirror before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from projects.evoseg.restart.prepare_foundation import atomic_json
from .public_mix import OFFICIAL_GREF_REVISION


def run(args):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, configure_http_backend, hf_hub_download
    import requests
    def backend():
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': args.proxy, 'https': args.proxy}
        return session
    configure_http_backend(backend_factory=backend)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).repo_info('FudanCVL/gRefCOCO', repo_type='dataset',
                                      revision=OFFICIAL_GREF_REVISION, files_metadata=True)
    inventory = []
    for file in info.siblings:
        if file.rfilename not in ('grefs(unc).json', 'instances.json'):
            continue
        if file.lfs is None:
            raise RuntimeError('official annotation lacks a published SHA-256')
        expected = file.lfs.sha256
        target = directory / file.rfilename
        existing = Path(args.existing) / ('grefs_unc.json' if file.rfilename.startswith('grefs') else file.rfilename)
        if not target.exists() and existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() == expected:
            temporary = target.with_suffix('.copying')
            shutil.copyfile(existing, temporary)
            temporary.replace(target)
        if not target.exists():
            hf_hub_download('FudanCVL/gRefCOCO', repo_type='dataset', revision=OFFICIAL_GREF_REVISION,
                            filename=file.rfilename, token=False, local_dir=directory)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected or target.stat().st_size != file.size:
            raise RuntimeError('official public annotation checksum mismatch')
        inventory.append({'file': file.rfilename, 'sha256': actual, 'bytes': file.size})
    if len(inventory) != 2:
        raise RuntimeError('official annotation inventory incomplete')
    report = {'repo': 'FudanCVL/gRefCOCO', 'revision': info.sha, 'byte_hashes_verified': True,
              'inventory': inventory, 'source_link': 'https://github.com/henghuiding/gRefCOCO',
              'generated_data': False}
    atomic_json(directory / 'SOURCE.json', report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proxy', default='http://127.0.0.1:17890')
    p.add_argument('--existing', default='/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco')
    p.add_argument('--output', default='/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco-official-pinned')
    run(p.parse_args())


if __name__ == '__main__':
    main()

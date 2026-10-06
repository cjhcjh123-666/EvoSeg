"""Fetch original COCO2014 instance GT; never replace it with pseudo masks."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import requests


# Same official bucket, valid HTTPS hostname. The images.cocodataset.org alias
# currently serves a certificate for a different name; never disable TLS.
URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
FILES = ('annotations/instances_train2014.json', 'annotations/instances_val2014.json')


def run(root, proxy):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'annotations_trainval2014.zip'
    if not archive.exists():
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': proxy, 'https': proxy}
        with session.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            expected = int(response.headers['Content-Length']) if 'Content-Length' in response.headers else None
            temporary = archive.with_suffix('.zip.incomplete')
            if temporary.exists():
                raise RuntimeError('existing partial COCO download; inspect before replacing it')
            total = 0
            with temporary.open('xb') as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
                    total += len(chunk)
            if expected is not None and total != expected:
                raise RuntimeError('incomplete COCO annotation download')
            temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for name in FILES:
            target = root / name
            if target.exists():
                if target.stat().st_size != source.getinfo(name).file_size:
                    raise RuntimeError('existing COCO annotation differs; do not overwrite it')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as reader, target.open('xb') as writer:
                for chunk in iter(lambda: reader.read(1 << 20), b''):
                    writer.write(chunk)
            print('Extracted original instance GT:', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014')
    parser.add_argument('--proxy', default='http://127.0.0.1:17890')
    args = parser.parse_args()
    run(args.root, args.proxy)


if __name__ == '__main__':
    main()

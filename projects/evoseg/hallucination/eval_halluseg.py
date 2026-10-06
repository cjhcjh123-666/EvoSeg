"""Four-combination public HalluSegBench evaluation with immutable pair caches."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from projects.evoseg.restart.prepare_foundation import atomic_json
from .protocol import (PROTOCOL, COMBINATIONS, SUBSETS, binary_union, model_fingerprint,
                       public_pairs, quartet_metrics)


def load_native_image_model(model_dir):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    model, loading = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True,
        local_files_only=True, output_loading_info=True)
    # Historical optional VIDEO gates are not an image prediction dependency.
    image_inactive = [key for key in loading.get('missing_keys', [])
                      if key.startswith(('existence_head.', 'temporal_existence_head.'))]
    problems = [key for key in loading.get('missing_keys', []) if key not in image_inactive]
    if problems or any(loading.get(k) for k in ('unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('incomplete native language/segmentation checkpoint loading: ' + str(loading))
    model = model.cuda().eval().requires_grad_(False)
    model.temporal_gate_enabled = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    return model, tokenizer, processor, loading, image_inactive


def run(args):
    import torch
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    pairs = public_pairs(args.data_root, args.limit)
    identity = model_fingerprint(args.model_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, processor, loading, image_inactive = load_native_image_model(args.model_dir)
    atomic_json(output / f'LOADING_rank{rank}.json', {'loading': loading, 'image_inactive_missing_keys': image_inactive,
                'model_fingerprint': identity, 'optional_video_gate_disabled': True})
    state = {'status': 'EVALUATING', 'rank': rank, 'pid': os.getpid(), 'world_size': world,
             'completed_pairs': 0, 'completed_predictions': 0, 'started_at_unix': time.time(),
             'model_fingerprint': identity, 'protocol': PROTOCOL, 'training_started': False}
    for pair_index in range(rank, len(pairs), world):
        pair = pairs[pair_index]
        directory = output / 'pairs' / pair['subset'] / f'{pair["index"]:06d}'
        destination = directory / 'CASE.json'
        if destination.exists():
            old = json.loads(destination.read_text())
            if (old['protocol'] != PROTOCOL or old['model_fingerprint'] != identity or
                    old['source_sha256'] != pair['source_sha256'] or len(old['predictions']) != 4):
                raise RuntimeError('prediction cache model, source or protocol mismatch')
            if not all((directory / (name + '.png')).exists() for name in COMBINATIONS):
                raise RuntimeError('incomplete pair mask files')
        else:
            directory.mkdir(parents=True, exist_ok=True)
            predictions, records = {}, {}
            for combination in COMBINATIONS:
                image_kind, query_kind = combination.split('_')
                with Image.open(pair[image_kind + '_image']) as original:
                    image = original.convert('RGB')
                query = pair[query_kind + '_query']
                with torch.inference_mode():
                    answer = model.predict_forward(image=image,
                        text=f'<image>\n Please segment {query} in this image.', tokenizer=tokenizer, processor=processor)
                mask, count = binary_union(answer['prediction_masks'], (image.height, image.width))
                temporary = directory / (combination + '.tmp')
                Image.fromarray(mask.astype(np.uint8) * 255).save(temporary, format='PNG')
                temporary.replace(directory / (combination + '.png'))
                predictions[combination] = mask
                records[combination] = {'native_answer': answer['prediction'], 'returned_mask_arrays': count,
                                        'foreground_pixels': int(mask.sum()), 'nonempty_mask': bool(mask.any()),
                                        'shape': list(mask.shape)}
            atomic_json(destination, {'subset': pair['subset'], 'index': pair['index'],
                'protocol': PROTOCOL, 'source_sha256': pair['source_sha256'], 'model_fingerprint': identity,
                'predictions': records, 'metrics': quartet_metrics(pair, predictions)})
        state.update(completed_pairs=state['completed_pairs'] + 1,
                     completed_predictions=state['completed_predictions'] + 4,
                     subset=pair['subset'], case=pair['index'], updated_at_unix=time.time(),
                     gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        atomic_json(output / f'STATUS_rank{rank}.json', state)
        print(json.dumps(state), flush=True)
    state.update(status='SHARD_COMPLETE', finished_at_unix=time.time())
    atomic_json(output / f'STATUS_rank{rank}.json', state)


def summarize(args):
    pairs = public_pairs(args.data_root, args.limit)
    identity = model_fingerprint(args.model_dir)
    accumulated = defaultdict(list)
    for pair in pairs:
        path = Path(args.output) / 'pairs' / pair['subset'] / f'{pair["index"]:06d}' / 'CASE.json'
        case = json.loads(path.read_text())
        if (case['model_fingerprint'] != identity or case['source_sha256'] != pair['source_sha256'] or
            case['protocol'] != PROTOCOL or set(case['predictions']) != set(COMBINATIONS)):
            raise RuntimeError('cannot summarize incomplete or incompatible predictions')
        accumulated[pair['subset']].append(case['metrics'])
    report = {'status': 'HALLUSEGBENCH_EVALUATION_COMPLETE', 'protocol': PROTOCOL,
              'pairs': len(pairs), 'predictions': len(pairs) * 4, 'model_dir': args.model_dir,
              'model_fingerprint': identity, 'full_public_evaluation': args.limit is None,
              'CMS_alpha': 3, 'subsets': {}, 'training_started': False, 'sota_claimed': False,
              'source_table_comparability': 'author metric formula checked; model-specific inference prompts require disclosure'}
    for subset, values in accumulated.items():
        report['subsets'][subset] = {'pairs': len(values), **{key: sum(v[key] for v in values) / len(values)
                                   for key in values[0]}}
    reason = accumulated['reason_seg_val'] + accumulated['reason_seg_test']
    report['reasoning_combined'] = {'pairs': len(reason), **{key: sum(v[key] for v in reason) / len(reason)
                                    for key in reason[0]}}
    if args.limit is None and report['predictions'] != 6956:
        raise RuntimeError('full public evaluation requires 1739 pairs / 6956 predictions')
    atomic_json(Path(args.output) / 'METRICS.json', report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='/9950backfile/chenjiahui/evo_artifacts/datasets/hallusegbench/test')
    p.add_argument('--limit', type=int, help='Pairs per subset: explicit diagnostic, not a benchmark')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('positive diagnostic limit required')
    (summarize if args.summarize else run)(args)


if __name__ == '__main__':
    main()

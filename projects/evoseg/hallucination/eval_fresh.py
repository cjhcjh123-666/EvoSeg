"""Fresh native SAMTok evaluation; learned adapter is actually used before decoding."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from projects.evoseg.restart.prepare_foundation import atomic_json
from projects.evoseg.dialog.runtime import NativeRuntime
from projects.evoseg.dialog.protocol import decode_annotation
from .interaction import InteractionGroundingAdapter
from .protocol import COMBINATIONS, PROTOCOL, public_pairs, quartet_metrics, read_mask
from .train_fresh import cached_records, prefix_inputs


def parse_native_codes(text):
    # Native author's flat-code extraction supports multiple output masks.
    ids = [int(x) for x in re.findall(r'<\|mt_(\d{4})\|>', text)]
    malformed = bool(len(ids) % 2)
    codes = []
    for index in range(0, len(ids) - 1, 2):
        first, second = ids[index], ids[index + 1]
        if not 0 <= first < 256 or not 256 <= second < 512:
            malformed = True
            continue
        codes.append([first, second - 256])
    if '<|mt_start|>' in text and not ids:
        malformed = True
    return codes, malformed


class FreshRuntime(NativeRuntime):
    def __init__(self, model_dir, assets, checkpoint=None):
        directory = Path(checkpoint) if checkpoint else None
        super().__init__(model_dir, assets, adapter_dir=directory / 'adapter' if directory else None)
        self.interface = None
        self.variant = None
        self.controls = {}
        self.checkpoint = checkpoint
        if directory:
            config = json.loads((directory / 'CONFIG.json').read_text())
            if config['old_faithful_inherited'] or config['generated_data']:
                raise RuntimeError('retired/self-generated route cannot enter this evaluator')
            self.variant = config['variant']
            self.interface = InteractionGroundingAdapter(config['language_dim'], config['prompt_dim']).cuda().eval()
            self.interface.load_state_dict(torch.load(directory / 'GROUNDING.pth', map_location='cpu', weights_only=True), strict=True)
            self.empty_threshold = config['empty_threshold']
            head_digest = hashlib.sha256((directory / 'GROUNDING.pth').read_bytes()).hexdigest()
            self.adapter_fingerprint += ':' + head_digest + ':' + self.variant + ':' + str(self.empty_threshold)

    @torch.inference_mode()
    def prefix_feature(self, image, query):
        inputs = prefix_inputs(self.processor, image, query).to(self.model.device)
        result = self.model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return result.hidden_states[-1][:, -1].float()

    @torch.inference_mode()
    def predict(self, image_path, query):
        # Public image/query only. No GT mask, target code or target ID input.
        self.set_image(image_path)
        inputs = prefix_inputs(self.processor, self.image, query).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False,
            return_dict_in_generate=True, output_hidden_states=self.interface is not None)
        answer = self.processor.decode(generated.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
        codes, malformed = parse_native_codes(answer)
        mask = np.zeros((self.image.height, self.image.width), dtype=bool)
        information = {'answer': answer, 'native_codes': codes, 'malformed_codes': malformed,
                       'head_used': self.interface is not None, 'empty_probability': None}
        if not codes:
            return mask, information
        views = None
        if self.interface is not None:
            full = generated.hidden_states[0][-1][:, -1].float()
            neutral = Image.new('RGB', self.image.size, (127, 127, 127))
            key = str(image_path)
            if key not in self.controls:
                visual = self.prefix_feature(self.image, 'an object')
                neither = self.prefix_feature(neutral, 'an object')
                self.controls = {key: (visual, neither)}
            visual, neither = self.controls[key]
            query_only = self.prefix_feature(neutral, query)
            if self.variant == 'full_view':
                visual, query_only, neither = visual * 0, query_only * 0, neither * 0
            views = (full, visual, query_only, neither)
        for pair in codes:
            values = torch.tensor([pair], device='cuda')
            embeddings = self.tokenizer.quantizer.embed_code(values)[:, None]
            embeddings = self.tokenizer.deconcate_quant_embed(embeddings)
            if self.interface is not None:
                result = self.interface(embeddings.reshape(1, -1), *views)
                empty_probability = float(result['empty_logit'].sigmoid())
                information['empty_probability'] = empty_probability
                if empty_probability >= self.empty_threshold:
                    return np.zeros_like(mask), information
                embeddings = result['grounding_prompt'][:, None]
            embeddings = embeddings.reshape(1, self.tokenizer.num_mask_tokens, -1)
            logits = self.tokenizer.model.inject_language_embd(self.sam_states, embeddings, nf_nobj=(1, 1))
            logits = F.interpolate(logits, size=mask.shape, mode='bilinear')
            mask |= (logits[0, 0] > .5).cpu().numpy()
        return mask, information


def evaluation_records(args):
    if args.benchmark == 'holdout':
        rows = [r for r in cached_records(args.cache) if r['holdout']]
        if args.limit is None:
            # Use every held-out original query; do not discard larger buckets.
            return sorted(rows, key=lambda r: r['id'])
        # Equal number per source bucket, without reading any benchmark scores.
        by_bucket = defaultdict(list)
        for row in rows:
            by_bucket[row['bucket']].append(row)
        each = args.limit // 3
        if each < 1:
            raise RuntimeError('empty diagnostic')
        return sorted([r for values in by_bucket.values() for r in values[:each]], key=lambda r: r['id'])
    if args.benchmark == 'gref_val':
        refs = json.loads((Path(args.gref_root) / 'grefs(unc).json').read_text())
        records = []
        for ref in refs:
            if ref['split'] != 'val':
                continue
            for sent in ref['sentences']:
                records.append({'id': f'{ref["ref_id"]}_{sent["sent_id"]}', 'query': sent['sent'],
                    'image': str(Path(args.coco_images) / ref['file_name']), 'image_id': ref['image_id'],
                    'private_annotation_ids': ref['ann_id'], 'private_no_target': ref['no_target']})
        return records if args.limit is None else records[:args.limit]
    pairs = public_pairs(args.hallu_root, args.limit)
    return [{**pair, 'id': f'{pair["subset"]}_{pair["index"]}'} for pair in pairs]


def metric_values(target, prediction, malformed, valid=None):
    if target.shape != prediction.shape:
        raise RuntimeError('prediction/GT shape mismatch')
    valid = np.ones_like(target, dtype=bool) if valid is None else valid
    if valid.shape != target.shape or not valid.any():
        raise RuntimeError('invalid evaluation ignore mask')
    intersection, union = int((target & prediction & valid).sum()), int(((target | prediction) & valid).sum())
    empty = not target.any()
    prediction_empty = not (prediction & valid).any()
    return {'intersection': intersection, 'union': union,
            'iou': intersection / union if union else 1., 'empty_gt': bool(empty),
            'predicted_empty': bool(prediction_empty), 'malformed': bool(malformed)}


def run(args):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    records = evaluation_records(args)
    runtime = FreshRuntime(args.model_dir, args.assets, args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotations = None
    if args.benchmark == 'gref_val':
        annotations = {int(a['id']): a for a in json.loads((Path(args.gref_root) / 'instances.json').read_text())['annotations']}
    started = time.time()
    for count, index in enumerate(range(rank, len(records), world), 1):
        record = records[index]
        destination = output / 'cases' / (record['id'] + '.json')
        if destination.exists():
            old = json.loads(destination.read_text())
            if old['model_identity'] != runtime.adapter_fingerprint or old['record_sha256'] != hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest():
                raise RuntimeError('stale/incompatible prediction cache')
        else:
            predictions, info = {}, {}
            if args.benchmark == 'halluseg':
                for combination in COMBINATIONS:
                    kind, query_kind = combination.split('_')
                    predictions[combination], info[combination] = runtime.predict(record[kind + '_image'], record[query_kind + '_query'])
                metrics = quartet_metrics(record, predictions)
            else:
                prediction, information = runtime.predict(record['image'], record['query'])
                if args.benchmark == 'holdout':
                    target = read_mask(record['human_gt_mask'])
                    valid = read_mask(record['human_valid_mask']) if record.get('human_valid_mask') else None
                else:
                    valid = None
                    target = np.zeros_like(prediction)
                    if not record['private_no_target']:
                        for annotation_id in record['private_annotation_ids']:
                            target |= decode_annotation(annotations[annotation_id], *target.shape, record['image_id'])
                metrics = metric_values(target, prediction, information['malformed_codes'], valid)
                predictions, info = {'mask': prediction}, {'mask': information}
            directory = output / 'masks' / record['id']
            directory.mkdir(parents=True, exist_ok=True)
            for name, mask in predictions.items():
                Image.fromarray(mask.astype(np.uint8) * 255).save(directory / (name + '.png'))
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, {'id': record['id'], 'model_identity': runtime.adapter_fingerprint,
                'record_sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                'benchmark': args.benchmark, 'subset': record.get('subset'), 'metrics': metrics, 'inference': info})
        atomic_json(output / f'STATUS_rank{rank}.json', {'status': 'EVALUATING', 'pid': os.getpid(),
                    'cases_done': count, 'benchmark': args.benchmark, 'updated_at_unix': time.time(),
                    'elapsed_seconds': time.time() - started})
    atomic_json(output / f'DONE_rank{rank}.json', {'status': 'SHARD_COMPLETE'})


def summarize(args):
    records = evaluation_records(args)
    cases = [json.loads((Path(args.output) / 'cases' / (r['id'] + '.json')).read_text()) for r in records]
    if len({case['model_identity'] for case in cases}) != 1:
        raise RuntimeError('mixed checkpoints in evaluation')
    report = {'cases': len(cases), 'benchmark': args.benchmark, 'checkpoint': args.checkpoint,
              'full_public_evaluation': args.limit is None and args.benchmark != 'holdout',
              'sota_claimed': False, 'model_identity': cases[0]['model_identity']}
    if args.benchmark == 'halluseg':
        for group, subsets in [('referring', ['refer_seg']), ('reasoning', ['reason_seg_val', 'reason_seg_test'])]:
            values = [c['metrics'] for c in cases if c['subset'] in subsets]
            report[group] = {'pairs': len(values), **{k: sum(v[k] for v in values) / len(values) for k in values[0]}}
    else:
        values = [c['metrics'] for c in cases]
        positive = [v for v in values if not v['empty_gt']]
        negative = [v for v in values if v['empty_gt']]
        if not positive or not negative:
            raise RuntimeError('both positive and no-target cases required')
        report.update(gIoU=sum(v['iou'] for v in values) / len(values),
                      cIoU=sum(v['intersection'] for v in values) / max(1, sum(v['union'] for v in values)),
                      positive_gIoU=sum(v['iou'] for v in positive) / len(positive),
                      N_acc=sum(v['predicted_empty'] for v in negative) / len(negative),
                      T_acc=sum(not v['predicted_empty'] for v in positive) / len(positive),
                      positive_cases=len(positive), no_target_cases=len(negative),
                      malformed_outputs=sum(v['malformed'] for v in values))
    atomic_json(Path(args.output) / 'METRICS.json', report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    p.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    p.add_argument('--gref-root', default=str(base / 'datasets/grefcoco-official-pinned'))
    p.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    p.add_argument('--hallu-root', default=str(base / 'datasets/hallusegbench/test'))
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--benchmark', choices=['holdout', 'gref_val', 'halluseg'], default='holdout')
    p.add_argument('--checkpoint')
    p.add_argument('--limit', type=int)
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    (summarize if args.summarize else run)(args)


if __name__ == '__main__':
    main()

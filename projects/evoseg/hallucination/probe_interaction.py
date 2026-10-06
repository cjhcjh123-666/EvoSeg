"""Public TRAIN-only image-disjoint mechanism probe, never a method benchmark."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from projects.evoseg.restart.prepare_foundation import atomic_json
from .eval_halluseg import load_native_image_model
from .interaction import factorial_interaction
from .protocol import digest_file, model_fingerprint


def select_train_cases(refs, image_count):
    forbidden = {r['file_name'] for r in refs if r['split'] != 'train'}
    groups = defaultdict(lambda: {False: [], True: []})
    for ref in refs:
        if ref['split'] != 'train' or ref['file_name'] in forbidden:
            continue
        empty = bool(ref['no_target'] or ref['ann_id'] == [-1])
        for sentence in ref['sentences']:
            if len(groups[ref['file_name']][empty]) < 2:
                groups[ref['file_name']][empty].append({'file_name': ref['file_name'],
                    'ref_id': ref['ref_id'], 'sent_id': sentence['sent_id'], 'query': sentence['sent'],
                    'private_gt_no_target': empty, 'public_split': 'train'})
    ready = [name for name, groups_for_image in sorted(groups.items())
             if len(groups_for_image[False]) == len(groups_for_image[True]) == 2]
    if len(ready) < image_count:
        raise RuntimeError('insufficient paired positive/no-target public TRAIN images')
    # One in four selected images is held out. Both query labels exist in each
    # image, preventing image identity alone from predicting the label.
    cases = []
    for index, image in enumerate(ready[:image_count]):
        for empty in (False, True):
            for record in groups[image][empty]:
                cases.append({**record, 'probe_holdout': index % 4 == 0})
    return cases


@torch.inference_mode()
def representation(model, processor, image, query):
    # Fixed SEG prefix is IDENTICAL for both classes and every neutral view.
    # No GT response, mask, annotation ID or no-target label enters this path.
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                {'type': 'text', 'text': f'Please segment {query} in this image.'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Sure, [SEG]'}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt',
                       min_pixels=model.min_pixels, max_pixels=model.max_pixels).to(model.device)
    seg_id = processor.tokenizer.convert_tokens_to_ids('[SEG]')
    positions = (inputs.input_ids[0] == seg_id).nonzero().flatten()
    if len(positions) != 1:
        raise RuntimeError('diagnostic SEG prefix is not uniquely token-aligned')
    output = model.model(**inputs, output_hidden_states=True, use_cache=False)
    feature = output.hidden_states[-1][0, positions.item()].float().cpu().clone()
    if not torch.isfinite(feature).all():
        raise RuntimeError('nonfinite native grounding representation')
    return feature, inputs.input_ids.shape[1], inputs.get('image_grid_thw').cpu().tolist()


def auc(labels, scores):
    positive, negative = scores[labels.bool()], scores[~labels.bool()]
    if len(positive) == 0 or len(negative) == 0:
        raise RuntimeError('probe must contain both public labels')
    return float(((positive[:, None] > negative).float() +
                  .5 * (positive[:, None] == negative).float()).mean())


def fit_probe(features, labels, holdout, seed=42):
    # Identical head capacity, steps, normalization and labels for all views.
    torch.manual_seed(seed)
    train = ~holdout
    center = features[train].mean(0)
    scale = features[train].std(0).clamp_min(1e-3)
    normalized = (features - center) / scale
    classifier = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=.1)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(normalized[train]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[train].float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = classifier(normalized[holdout]).squeeze(-1).sigmoid()
    selected = labels[holdout].bool()
    decisions = scores >= .5
    return {'holdout_cases': int(holdout.sum()), 'empty_target_AUROC': auc(labels[holdout], scores),
            'balanced_accuracy_at_half': float(((decisions[selected]).float().mean() +
                                                (~decisions[~selected]).float().mean()) / 2),
            'predicted_empty_fraction': float(decisions.float().mean()),
            'optimizer_updates': 100}


def run(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'PROBE_REPORT.json').exists():
        raise RuntimeError('existing diagnostic evidence cannot be overwritten')
    refs = json.loads(Path(args.grefs).read_text())
    cases = select_train_cases(refs, args.images)
    model, _, processor, loading, inactive = load_native_image_model(args.model_dir)
    rows, shared = [], {}
    started = time.time()
    for index, record in enumerate(cases):
        with Image.open(Path(args.coco_images) / record['file_name']) as original:
            image = original.convert('RGB')
        neutral = Image.new('RGB', image.size, (127, 127, 127))
        full, length, grid = representation(model, processor, image, record['query'])
        query_only, neutral_length, neutral_grid = representation(model, processor, neutral, record['query'])
        if length != neutral_length or grid != neutral_grid:
            raise RuntimeError('neutral visual control changed token geometry')
        if record['file_name'] not in shared:
            visual_only, _, _ = representation(model, processor, image, 'an object')
            neither, _, _ = representation(model, processor, neutral, 'an object')
            shared[record['file_name']] = (visual_only, neither)
        visual_only, neither = shared[record['file_name']]
        rows.append({'full': full, 'visual_only': visual_only, 'query_only': query_only,
                     'interaction': factorial_interaction(full, visual_only, query_only, neither)})
        atomic_json(output / 'PROBE_STATUS.json', {'status': 'EXTRACTING_PUBLIC_TRAIN_FEATURES',
                    'completed_cases': len(rows), 'planned_cases': len(cases),
                    'elapsed_seconds': time.time() - started, 'updated_at_unix': time.time(),
                    'new_method_training_started': False})
    features = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    labels = torch.tensor([r['private_gt_no_target'] for r in cases])
    holdout = torch.tensor([r['probe_holdout'] for r in cases])
    reports = {key: fit_probe(values, labels, holdout) for key, values in features.items()}
    torch.save({'features': features, 'labels': labels, 'holdout': holdout}, output / 'TRAIN_FEATURES.pth')
    atomic_json(output / 'CASE_PROVENANCE.json', {'source_sha256': digest_file(args.grefs), 'cases': cases})
    report = {'status': 'PUBLIC_TRAIN_INTERACTION_PROBE_COMPLETE', 'cases': len(cases), 'images': args.images,
              'source_split': 'public_gRefCOCO_train_only', 'image_disjoint_probe_split': True,
              'official_val_test_images_excluded': True, 'ground_truth_in_inference_inputs': False,
              'neutral_visual': 'constant RGB 127 preserving dimensions/token grid',
              'neutral_query': 'an object', 'prefix': 'fixed Sure SEG, independent of GT label',
              'probes': reports, 'model_fingerprint': model_fingerprint(args.model_dir),
              'source_checkpoint_pretraining_exposure_possible': True,
              'new_method_training_started': False, 'method_integrated_with_SAM': False,
              'benchmark_result': False, 'sota_claimed': False, 'loading': loading,
              'image_inactive_missing_keys': inactive,
              'next_gate': 'interaction must beat same-budget full/query-only controls; inspect neutral-view shift before SAM integration'}
    atomic_json(output / 'PROBE_REPORT.json', report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    p.add_argument('--model-dir', default=str(base / 'models/EvoSeg-Qwen3-VL-4B-Faithful'))
    p.add_argument('--grefs', default=str(base / 'datasets/grefcoco/grefs_unc.json'))
    p.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    p.add_argument('--images', type=int, default=32)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.images < 8 or args.images % 4:
        p.error('image count must be >=8 and divisible by four')
    run(args)


if __name__ == '__main__':
    main()

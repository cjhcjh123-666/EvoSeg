"""Complete public HalluSegBench inputs and author-equivalent pixel metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

PROTOCOL = 'halluseg_public_four_combinations_alpha3_v1'
SUBSETS = {'refer_seg': 1340, 'reason_seg_val': 74, 'reason_seg_test': 325}
COMBINATIONS = ('factual_factual', 'factual_counterfactual',
                'counterfactual_factual', 'counterfactual_counterfactual')


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_asset(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('public asset path escapes dataset directory')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def public_pairs(root, limit=None):
    result = []
    for subset, expected in SUBSETS.items():
        directory = Path(root) / subset
        source = directory / 'data.json'
        records = json.loads(source.read_text())
        if len(records) != expected:
            raise RuntimeError(f'{subset}: expected {expected} public pairs, got {len(records)}')
        source_digest = digest_file(source)
        for index, record in enumerate(records if limit is None else records[:limit]):
            factual_query = record['factual_label'] if subset == 'refer_seg' else record['question']
            cf_query = record['counterfactual_label'] if subset == 'refer_seg' else record['counterfactual_question']
            cf_mask = record.get('counterfactual_mask_path', record.get('counterfactual_mask'))
            if not factual_query or not cf_query or not cf_mask:
                raise ValueError('incomplete public query or mask metadata')
            pair = {'subset': subset, 'index': index, 'source_sha256': source_digest,
                    'factual_query': factual_query, 'counterfactual_query': cf_query}
            for key, relative in [('factual_image', record['factual_image_path']),
                                  ('counterfactual_image', record['counterfactual_image_path']),
                                  ('factual_mask', record['factual_mask_path']), ('counterfactual_mask', cf_mask)]:
                pair[key] = str(safe_asset(directory, relative))
            result.append(pair)
    return result


def binary_union(masks, shape):
    """Array count and SEG-token count do not establish nonempty prediction."""
    union = np.zeros(shape, dtype=bool)
    count = 0
    for mask in masks:
        if hasattr(mask, 'detach'):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError('native predicted mask has wrong shape or nonfinite values')
        if not np.isin(mask, [0, 1]).all():
            raise ValueError('requires native binary masks, not an invented logit threshold')
        mask = mask.astype(bool)
        union |= mask
        count += 1
    return union, count


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError('cannot read public mask: ' + str(path))
    return mask > 0


def resized(mask, shape):
    if mask.shape == tuple(shape):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(target, prediction, shape=None):
    shape = target.shape if shape is None else shape
    target, prediction = resized(target, shape), resized(prediction, shape)
    union = int((target | prediction).sum())
    # Author's HalluSegBench consistency convention is 0 for empty union.
    return int((target & prediction).sum()) / union if union else 0.


def confusion_mask_score(target_region, false_prediction, alpha=3.):
    false_prediction = resized(false_prediction, target_region.shape)
    confusing = int((target_region & false_prediction).sum())
    unrelated = int((~target_region & false_prediction).sum())
    denominator = alpha * int(target_region.sum())
    return (alpha * confusing + unrelated) / denominator if denominator else 0.


def quartet_metrics(pair, predictions):
    if set(predictions) != set(COMBINATIONS):
        raise ValueError('must score all four public image-query combinations')
    original = read_mask(pair['factual_mask'])
    edited = read_mask(pair['counterfactual_mask'])
    factual_iou = iou(original, predictions['factual_factual'])
    textual_overlap = iou(original, predictions['factual_counterfactual'])
    # Match get_consistency.py: visual-counterfactual comparison uses edited
    # GT, with BOTH masks nearest-resized to factual GT dimensions.
    visual_overlap = iou(edited, predictions['counterfactual_factual'], original.shape)
    cms_fact = confusion_mask_score(original, predictions['factual_counterfactual'])
    cms_cf = confusion_mask_score(edited, predictions['counterfactual_factual'])
    return {'factual_IoU': factual_iou,
            'counterfactual_positive_IoU': iou(edited, predictions['counterfactual_counterfactual']),
            'delta_IoU_textual': factual_iou - textual_overlap,
            'delta_IoU_visual': factual_iou - visual_overlap,
            'CMS_factual': round(cms_fact, 4), 'CMS_counterfactual': round(cms_cf, 4),
            'CMS_factual_unrounded': cms_fact, 'CMS_counterfactual_unrounded': cms_cf,
            'false_acceptance_factual_image': int(predictions['factual_counterfactual'].any()),
            'false_acceptance_counterfactual_image': int(predictions['counterfactual_factual'].any())}


def model_fingerprint(directory):
    directory = Path(directory)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    inventory = []
    for name in sorted(set(index['weight_map'].values())):
        stat = (directory / name).stat()
        if stat.st_size == 0:
            raise RuntimeError('empty model weight shard')
        inventory.append((name, stat.st_size, stat.st_mtime_ns))
    configuration = {p.name: digest_file(p) for p in directory.iterdir()
                     if p.suffix in ('.py', '.json', '.jinja') and p.is_file()}
    # Inventory/version identity; explicitly not a byte SHA of the weight data.
    identity = {'path': str(directory.resolve()), 'weight_stat_inventory': inventory,
                'configuration_sha256': configuration}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

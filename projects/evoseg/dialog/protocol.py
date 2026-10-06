"""History isolation and exact foreground metrics for reconstructed native MRSeg."""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
from pycocotools import mask as mask_utils

from .public_data import IMAGE, REFERENCE
from .runtime import token_string


PROTOCOL_VERSION = 'mrseg_native_reconstruction_v1_mask_and_box_to_vq'


def decode_annotation(annotation, height, width, image_id=None):
    if image_id is not None and int(annotation['image_id']) != image_id:
        raise ValueError('GT annotation belongs to a different source image')
    segmentation = annotation['segmentation']
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle['counts'], list):
            rle = mask_utils.frPyObjects(rle, height, width)
    elif segmentation:
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    else:
        return np.zeros((height, width), dtype=bool)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.any(2)
    if mask.shape != (height, width):
        raise ValueError('GT mask/image shape mismatch')
    return mask.astype(bool)


def history_reference_codes(turn, annotation_id, history, mode, gt_lookup):
    if mode == 'gt_history':
        return gt_lookup(annotation_id)
    if mode != 'predicted_history':
        raise ValueError('unknown history protocol')
    if turn['external_visual_references']:
        # A user-provided cue, not previous GT output. Excluded from main val.
        raise ValueError('external visual cues need a separately declared evaluation')
    if len(turn['reference_rounds']) != 1:
        raise ValueError('closed-loop reference requires one explicit public round pointer')
    index = turn['reference_rounds'][0] - 1
    if index < 0 or index >= len(history):
        raise ValueError('reference points outside generated history')
    return history[index]['codes']


def render_query(turn, history, mode, gt_lookup):
    def replace(match):
        if match.group(0).startswith('[BOX-ENCODE:'):
            # Native VQ reference encoding includes the normalized bbox along
            # with the mask. Do not invent a language-level bbox format.
            return ''
        annotation_id = int(match.group(1).split('|')[-1])
        codes = history_reference_codes(turn, annotation_id, history, mode, gt_lookup)
        return token_string(codes) if codes is not None else ' [no valid previous mask] '
    return REFERENCE.sub(replace, IMAGE.sub('', turn['query'])).strip()


def assistant_history(mode, generated_text, generated_codes, target_codes):
    if mode == 'predicted_history':
        # The target code callback is never evaluated in this branch.
        # apply_chat_template adds its own turn terminator; preserve mask tokens
        # but do not embed a second EOS in the preceding assistant's text.
        return generated_text.removesuffix('<|im_end|>').rstrip(), generated_codes
    if mode != 'gt_history':
        raise ValueError('unknown history protocol')
    codes = target_codes()
    value = token_string(codes) if codes is not None else None
    return '```json\n' + json.dumps([{'mask_2d': value}]) + '\n```', codes


class RoundMetrics:
    def __init__(self):
        self.rounds = defaultdict(lambda: {'count': 0, 'intersection': 0, 'union': 0,
                                           'iou_sum': 0., 'invalid_outputs': 0})

    def add(self, round_id, target, prediction, invalid=False):
        if target.shape != prediction.shape:
            raise ValueError('metric mask shapes differ')
        target, prediction = target.astype(bool), prediction.astype(bool)
        intersection, union = int((target & prediction).sum()), int((target | prediction).sum())
        item = self.rounds[int(round_id)]
        item['count'] += 1
        item['intersection'] += intersection
        item['union'] += union
        item['iou_sum'] += intersection / union if union else 1.
        item['invalid_outputs'] += int(invalid)

    def summary(self):
        return {str(round_id): {**item, 'cIoU': item['intersection'] / item['union'] if item['union'] else 1.,
                               'gIoU': item['iou_sum'] / item['count']}
                for round_id, item in sorted(self.rounds.items())}

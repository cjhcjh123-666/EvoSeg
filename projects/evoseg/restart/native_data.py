"""Public MeViS data, native SaSaSa2VA compression and teacher forcing.

No generated expressions or labels. The training sampling, 3x3 grid, ten
keyframe masks and assistant Clip_N [SEG] format follow the upstream dataset.
The held-out videos are excluded from this fine-tune, not claimed unseen by
the released foundation's pretraining.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools import mask as mask_utils


def split_video_ids(videos, seed=42, fraction=.05):
    if not 0 < fraction < 1:
        raise ValueError('held-out fraction must be in (0, 1)')
    ids = sorted(videos)
    if len(ids) < 2:
        raise ValueError('need at least two videos for disjoint split')
    ranked = sorted(ids, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    held_out = set(ranked[:count])
    return [key for key in ids if key not in held_out], sorted(held_out)


def native_train_indices(length):
    if length < 1:
        raise ValueError('empty video')
    return sorted(index % length for index in range(100))


def encode_turns(tokenizer, template, expressions, image_tokens, rng):
    ids, labels = [], []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(-100)
    questions = (
        'Please segment {class_name} in this image.',
        'Can you segment the {class_name} in this image?',
        'Please identify and segment the {class_name} in this image.',
        'Where is the {class_name} in this picture? Please respond with a segmentation mask.',
    )
    clips = ', '.join(f'Clip_{i} [SEG]' for i in range(1, 11)) + '.'
    answers = ('It is ', 'Sure, ', 'Sure, it is ', 'Sure, the segmentation result is ', '')
    for turn, expression in enumerate(expressions):
        query = expression if '?' in expression else rng.choice(questions).format(
            class_name=expression.replace('.', '').strip().lower())
        if turn == 0:
            query = image_tokens + '\n' + query
        prompt = template['INSTRUCTION'].format(input=query, round=turn + 1, bot_name='BOT')
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(rng.choice(answers) + clips + template['SUFFIX'],
                                      add_special_tokens=False)
        ids.extend(prompt_ids)
        labels.extend([-100] * len(prompt_ids))
        ids.extend(answer_ids)
        labels.extend(answer_ids)
        sep = tokenizer.encode(template.get('SEP', ''), add_special_tokens=False)
        ids.extend(sep)
        labels.extend([-100] * len(sep))
    return torch.tensor(ids), torch.tensor(labels)


def decode_union(mask_dict, anno_ids, frame_index, shape):
    target = np.zeros(shape, dtype=np.uint8)
    for anno in anno_ids:
        annotations = mask_dict[str(anno)]
        if frame_index >= len(annotations):
            raise ValueError('annotation frame count does not cover the video')
        encoded = annotations[frame_index]
        if encoded is not None:
            decoded = mask_utils.decode(encoded)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            if decoded.shape != shape:
                raise ValueError('annotation and frame dimensions differ')
            target |= decoded.astype(np.uint8)
    return target


class PublicMevisNative(Dataset):
    def __init__(self, root, tokenizer, template, *, seed=42, expressions_per_video=5):
        self.root = Path(root)
        self.videos = json.loads((self.root / 'meta_expressions_v2.json').read_text())['videos']
        self.mask_dict = json.loads((self.root / 'mask_dict.json').read_text())
        self.train_ids, self.held_out_ids = split_video_ids(self.videos, seed)
        self.tokenizer, self.template = tokenizer, template
        self.seed, self.epoch = seed, 0
        self.expressions_per_video = expressions_per_video
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        video_id = self.train_ids[index]
        metadata = self.videos[video_id]
        frames = sorted(metadata['frames'])
        indices = native_train_indices(len(frames))
        rng = random.Random(f'{self.seed}:{self.epoch}:{video_id}')
        expression_ids = sorted(metadata['expressions'])
        # Upstream samples with replacement, including videos with >5 queries.
        selected = [rng.choice(expression_ids) for _ in range(self.expressions_per_video)]
        expressions = [metadata['expressions'][key] for key in selected]
        image_cache = {}
        pixels, grounding, targets = [], [], []
        for offset in range(0, 100, 10):
            clip = []
            for frame_index in indices[offset:offset + 10]:
                if frame_index not in image_cache:
                    with Image.open(self.root / 'JPEGImages' / video_id / (frames[frame_index] + '.jpg')) as im:
                        image_cache[frame_index] = im.convert('RGB')
                clip.append(image_cache[frame_index])
            width, height = clip[0].size
            if any(image.size != (width, height) for image in clip):
                raise ValueError('inconsistent frame dimensions')
            grid = Image.new('RGB', (3 * width, 3 * height))
            for tile, image in enumerate(clip[1:]):
                grid.paste(image, ((tile % 3) * width, (tile // 3) * height))
            pixels.extend([self.transform(clip[0]), self.transform(grid)])
            grounding.append(torch.from_numpy(np.asarray(clip[0].resize((1024, 1024))).copy()).permute(2, 0, 1))
            targets.append(torch.stack([
                torch.from_numpy(decode_union(self.mask_dict, expression['anno_id'],
                                             indices[offset], (height, width)))
                for expression in expressions]))
        image_token = '<img>' + '<IMG_CONTEXT>' * 256 + '</img>'
        image_tokens = '\n'.join([image_token] * 20)
        ids, labels = encode_turns(self.tokenizer, self.template,
                                  [expression['exp'] for expression in expressions], image_tokens, rng)
        if len(ids) > 12288:
            raise ValueError('native context exceeded; never truncate SEG or image tokens silently')
        if (ids == self.tokenizer.convert_tokens_to_ids('[SEG]')).sum() != 10 * len(expressions):
            raise ValueError('SEG tokenizer or prompt contract mismatch')
        if (ids == self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')).sum() != 5120:
            raise ValueError('native image-token count mismatch')
        return {'input_ids': ids, 'labels': labels, 'pixel_values': torch.stack(pixels),
                'grounding_pixels': torch.stack(grounding), 'targets': torch.stack(targets),
                'video_id': video_id, 'expression_ids': selected}

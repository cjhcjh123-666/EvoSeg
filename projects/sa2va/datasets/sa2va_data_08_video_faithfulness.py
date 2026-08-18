"""Video faithfulness (temporal absence / identity / global absence) dataset.

Reads the manifest produced by
projects/evoseg/tools/build_video_faithfulness_train.py and emits Sa2VA video
samples that teach frame-wise existence:

  temporal_absence / identity_swap : real query + "Sure, [SEG]" answer with the
      referenced instance's GT masks (ZERO on absent frames). The mask loss on
      zero frames teaches the decoder to STOP propagating after disappearance
      and not to jump to lookalike instances.
  global_absence                   : mismatched cross-video query + refusal
      answer with all-zero masks (the no-[SEG] path attaches a zero embedding,
      same as the image no-target fix in Sa2VA.forward).

The frame sampler guarantees the presence->absence boundary is visible so the
"stop" signal is present in every temporal sample.
"""
import json
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image

from .common import SEG_QUESTIONS, ANSWER_LIST
from .base import Sa2VABaseDataset


class Sa2VA08VideoFaithfulnessDataset(Sa2VABaseDataset):

    def __init__(self,
                 manifest_file,
                 image_folder,
                 ann_folder,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 tokenizer=None,
                 max_length=4096,
                 arch_type='qwen',
                 preprocessor=None,
                 repeats=1,
                 sampled_frames=8,
                 name='VideoFaithfulnessDataset',
                 **kwargs):
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
            **kwargs,
        )
        self.image_folder = image_folder
        self.ann_folder = ann_folder
        self.sampled_frames = sampled_frames
        self.cases = json.load(open(manifest_file))['cases']

    def real_len(self):
        return len(self.cases)

    @property
    def modality_length(self):
        return [self._get_modality_length_default() for _ in range(len(self))]

    def __len__(self):
        return self.real_len() * self.repeats

    def __getitem__(self, index):
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data
        raise RuntimeError(f'Failed to get valid data after '
                           f'{self._max_refetch + 1} attempts')

    def _sample_frames(self, frames, presence, T):
        """Sample `sampled_frames` frames preserving the presence boundary."""
        if len(frames) <= self.sampled_frames:
            idxs = list(range(len(frames)))
        else:
            idxs = list(np.linspace(0, len(frames) - 1,
                                    self.sampled_frames).round().astype(int))
            idxs = sorted(set(idxs))
            idxs = [i for i in idxs if i < len(frames)]
            # ensure the disappearance / appearance boundary is visible
            if T > 0 and np.any(presence):
                t0 = int(np.where(presence)[0][0])
                t1 = int(np.where(presence)[0][-1])
                if t1 < T - 1 and (T - 1) not in idxs:
                    idxs.append(T - 1)
                if t0 > 0 and 0 not in idxs:
                    idxs.append(0)
                idxs = sorted(set(idxs))
        return idxs

    def prepare_data(self, index):
        index = index % self.real_len()
        c = self.cases[index]
        vid = c['video_id']
        frames = c['frames']
        T = len(frames)
        presence = np.array(c['presence'], dtype=bool)
        query = c['query'].lower().replace('.', '').strip()

        idxs = self._sample_frames(frames, presence, T)
        sel_frames = [frames[i] for i in idxs]

        # load images
        images = []
        for f in sel_frames:
            p = os.path.join(self.image_folder, vid, f + '.jpg')
            im = Image.open(p).convert('RGB')
            if im is None:
                return None
            images.append(im)

        # conversation
        question = random.choice(SEG_QUESTIONS).format(class_name=query)
        if c['category'] in ('temporal_absence', 'identity_swap'):
            answer = random.choice(ANSWER_LIST)          # contains [SEG]
        else:
            answer = f"I don't see {query} in this video."
        conversation = [
            {'from': 'human', 'value': '<image>\n' + question},
            {'from': 'gpt', 'value': answer},
        ]

        # masks (F, H, W)
        H, W = images[0].size[1], images[0].size[0]
        masks = np.zeros((len(sel_frames), H, W), dtype=np.uint8)
        if c['instance'] is not None:
            inst = int(c['instance'])
            for j, f in enumerate(sel_frames):
                p = os.path.join(self.ann_folder, vid, f + '.png')
                if not os.path.isfile(p):
                    continue
                a = np.array(Image.open(p).convert('L'))
                masks[j] = (a == inst).astype(np.uint8)

        out_data_dict = {}
        out_data_dict['masks'] = torch.from_numpy(masks)

        try:
            image_data = self._process_multiple_images(images)
            out_data_dict.update(image_data)
            num_frames = len(images)
            image_token_str = self._create_token_string(
                image_data['num_image_tokens'], num_frames)
            conversations = self._process_conversations_for_encoding(
                conversation, image_token_str, is_video=True)
            if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data:
                conversations = self._expand_video_tokens(
                    conversations, image_data['num_frame_tokens'],
                    image_data['num_image_tokens'])
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
        except Exception as e:
            print(f'Error processing video {vid}: {e}', flush=True)
            return None

        out_data_dict['type'] = 'video'
        return out_data_dict

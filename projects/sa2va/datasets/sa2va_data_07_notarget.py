"""No-target (abstain) dataset for EvoSeg faithful segmentation.

Reads a manifest produced by tools/build_no_target_abstain.py:

  {"image": <file_name>, "text": [query], "response": "<abstain text>"}

Emits a human question (same SEG_QUESTIONS wording as present data) followed
by the abstain response. NO masks are attached: the model's forward then takes
the pseudo-zero-mask path (seg_valid=False), which reinforces empty masks and
keeps the SAM2 decoder training signal consistent with "no target".
"""
import random
from typing import List

import torch

from .common import SEG_QUESTIONS
from .base import Sa2VABaseDataset

from third_parts.mmdet.datasets.refcoco import RefCocoDataset

import mmengine


class Sa2VA07NoTargetDataset(RefCocoDataset, Sa2VABaseDataset):

    def __init__(self,
                 data_root,
                 ann_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='images/'),
                 tokenizer=None,
                 max_length=2048,
                 single_image_mode=False,
                 arch_type: str = 'qwen',
                 preprocessor=None,
                 repeats: int = 1,
                 name: str = 'NoTargetDataset',
                 **kwargs):

        RefCocoDataset.__init__(self,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file='',
            **kwargs,
        )
        Sa2VABaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
        )
        self.begin_str = '<image>\n'
        self.image_folder = data_root
        self.single_image_mode = single_image_mode

    def load_data_list(self) -> List[dict]:
        self.annotations = mmengine.load(self.ann_file, file_format='json')
        img_prefix = self.data_prefix['img_path']
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        data_list = []
        for item in self.annotations:
            data_list.append({
                'img_path': join_path(img_prefix, item['image']),
                'text': item['text'],
                'response': item['response'],
            })
        if len(data_list) == 0:
            raise ValueError(f'No sample in no-target split.')
        return data_list

    @property
    def modality_length(self):
        return [self._get_modality_length_default() for _ in range(len(self))]

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        conversation = []
        for phrase, resp in zip(ann_info['text'], [ann_info['response']]):
            phrase = phrase.lower()
            if phrase.endswith('.'):
                phrase = phrase[:-1]
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if len(conversation) == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': resp})

        # Emit a zero mask so batches can mix no-target and present samples
        # (collate keys masks per sample). With no [SEG] in the answer the
        # forward attaches a zero embedding whose GT is this zero mask,
        # reinforcing empty masks (abstention).
        ann_info.update({
            'masks': torch.zeros((1, height, width), dtype=torch.uint8),
            'conversations': conversation,
            'image': image_path,
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self._read_image(image_file)
            if image is None:
                return None
            image_data = self._process_single_image(image, self.single_image_mode)
            out_data_dict.update(image_data)
            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
            conversation = self._process_conversations_for_encoding(
                data_dict['conversations'], image_token_str)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
        else:
            conversation = self._process_conversations_for_encoding(
                data_dict['conversations'], None)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
        return out_data_dict

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len() * self.repeats

    # !DO NOT CHANGE!
    # Re-write __getitem__ to override the default multi-inheritance behavior
    # (same pattern as Sa2VA01RefSeg / Sa2VABaseDataset): mmengine's
    # BaseDataset.__getitem__ would otherwise win the MRO and bypass the
    # repeats index mapping.
    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
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

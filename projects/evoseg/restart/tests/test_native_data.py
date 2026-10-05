import random

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from projects.evoseg.restart.native_data import decode_union, encode_turns, native_train_indices, split_video_ids


def test_split_disjoint_reproducible_and_input_order_independent():
    ids = [f'vid{i}' for i in range(100)]
    train, held = split_video_ids(ids)
    assert len(held) == 5
    assert set(train).isdisjoint(held)
    assert set(train) | set(held) == set(ids)
    assert (train, held) == split_video_ids(reversed(ids))


def test_native_wrap_compression_indices_not_uniform_inference():
    assert native_train_indices(150) == list(range(100))
    indices = native_train_indices(12)
    assert len(indices) == 100 and indices == sorted(indices)
    assert indices[-1] == 11
    with pytest.raises(ValueError):
        native_train_indices(0)


def test_union_masks_and_public_empty_target():
    shape = (3, 4)
    a = np.zeros(shape, dtype=np.uint8)
    a[0, 0] = 1
    b = np.zeros(shape, dtype=np.uint8)
    b[2, 3] = 1
    masks = {'a': [mask_utils.encode(np.asfortranarray(a))],
             'b': [mask_utils.encode(np.asfortranarray(b))]}
    assert np.array_equal(decode_union(masks, ['a', 'b'], 0, shape), a | b)
    assert decode_union(masks, [], 0, shape).sum() == 0
    with pytest.raises(ValueError, match='frame count'):
        decode_union(masks, ['a'], 1, shape)


class CharacterTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 10 for char in text]


def test_teacher_forcing_labels_only_assistant_outputs():
    tokenizer = CharacterTokenizer()
    template = {'INSTRUCTION': 'USER:{input}\nASSISTANT:', 'SUFFIX': '<END>', 'SEP': '\n'}
    ids, labels = encode_turns(tokenizer, template, ['dog', 'cat'], 'IMAGE', random.Random(42))
    output = ''.join(chr(int(x) - 10) for x in labels if int(x) != -100)
    assert output.count('[SEG]') == 20
    assert 'USER' not in output and 'IMAGE' not in output and 'dog' not in output
    assert (ids[labels != -100] == labels[labels != -100]).all()

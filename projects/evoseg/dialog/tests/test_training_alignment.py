from types import SimpleNamespace
import json

import pytest

import torch
from torch import nn

from projects.evoseg.dialog.train_public_pilot import PublicAlignment, continuation_manifest
from projects.evoseg.dialog.pilot_data import SPLIT_VERSION


class FakeCausalLanguage(nn.Module):
    """Test routing only; not a model result or a test of VLM causality."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = nn.Embedding(16, 8)
        self.future_offset = 0.

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels, use_cache, output_hidden_states):
        assert not use_cache
        hidden = self.embedding(input_ids).clone()
        hidden[:, 2:] += self.future_offset
        return SimpleNamespace(loss=hidden.sum() * 0 + 1., hidden_states=[hidden])


def test_auxiliary_query_does_not_read_current_answer_positions():
    torch.manual_seed(42)
    language = FakeCausalLanguage()
    model = PublicAlignment(language, auxiliary=True)
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4]])}
    labels = torch.tensor([[-100, -100, 3, 4]])
    history = torch.tensor([[[5, 6]]])
    valid = torch.tensor([[True]])
    pixels = torch.randn(1, 256, 2, 2)
    target = torch.zeros(1, 2, 2)
    pointer = torch.tensor([1, 0])
    _, before = model(inputs, labels, 2, history, valid, pixels, target, pointer)
    language.future_offset = 1000.
    _, after = model(inputs, labels, 2, history, valid, pixels, target, pointer)
    for key in ('witness', 'operation', 'scope'):
        torch.testing.assert_close(before[key], after[key], atol=0., rtol=0.)


def test_continuation_uses_total_updates_and_preserves_split(tmp_path):
    records = [{'image': '/images/a.jpg', 'foundation_revision': 'test-revision'}]
    manifest = {'variant': 'plain_lora', 'seed': 42, 'effective_batch': 8, 'split': SPLIT_VERSION,
                'foundation_revision': 'test-revision', 'generated_drafts_included': False,
                'editing_branch_supervised': False, 'status': 'PUBLIC_PILOT_TRAINING_COMPLETE',
                'train_images': ['/images/a.jpg'], 'optimizer_updates': 200, 'cumulative_model_updates': 264}
    (tmp_path / 'TRAINING_MANIFEST.json').write_text(json.dumps(manifest))
    (tmp_path / 'adapter').mkdir()
    (tmp_path / 'adapter/adapter_model.safetensors').write_bytes(b'unit test existence fixture only')
    _, initial = continuation_manifest(tmp_path, 'plain_lora', 42, 8, records)
    assert initial == 264
    with pytest.raises(RuntimeError, match='incompatible'):
        continuation_manifest(tmp_path, 'plain_lora', 42, 16, records)
    with pytest.raises(RuntimeError, match='images differ'):
        continuation_manifest(tmp_path, 'plain_lora', 42, 8,
                              [{'image': '/images/other.jpg', 'foundation_revision': 'test-revision'}])
    with pytest.raises(RuntimeError, match='incompatible'):
        continuation_manifest(tmp_path, 'witness_scope_aux', 42, 8, records)


def test_no_checkpoint_is_not_a_resume():
    assert continuation_manifest(None, 'plain_lora', 42, 8, []) == (None, 0)

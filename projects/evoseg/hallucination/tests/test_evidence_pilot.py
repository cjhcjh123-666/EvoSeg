from collections import Counter

import pytest
import torch

from projects.evoseg.hallucination.evidence_pilot import pack_inputs, select_training


def test_inputs_exclude_human_labels_and_empty_outputs_are_retained():
    semantic = torch.randn(1, 8)
    proposal = {'parent_empty_logit': torch.tensor([-4.]), 'prompt': torch.randn(1, 4),
                'parent_logits': torch.randn(1, 1, 2, 2), 'human_target_precision': torch.tensor([.9])}
    a = pack_inputs([proposal], semantic, .5)
    proposal['human_target_precision'].fill_(0)
    b = pack_inputs([proposal], semantic, .5)
    assert set(a) == {'semantic', 'native_accepted', 'candidates'}
    assert set(a['candidates'][0]) == {'prompt', 'candidate_logits'}
    torch.testing.assert_close(a['semantic'], b['semantic'])
    assert a['native_accepted'].item()
    empty = pack_inputs([], semantic, .5)
    assert not empty['native_accepted'].item() and empty['candidates'] == []


def test_large_budget_reuses_all_original_training_buckets_without_holdout():
    quotas = {'ref_positive': 448, 'gref_positive': 224, 'gref_empty': 256, 'reason_positive': 96}
    rows = []
    for kind, count in quotas.items():
        for i in range(count):
            rows.append({'id': f'{kind}_{i:04d}', 'bucket': kind, 'holdout': False,
                'native_proposals': int(i >= 3), 'source_split': 'train',
                'generated_query': False, 'pseudo_label': False})
    rows.append({**rows[0], 'id': 'heldout', 'holdout': True})
    selected = select_training(rows, 1024)
    assert dict(Counter(r['bucket'] for r in selected)) == quotas
    assert all(not r['holdout'] for r in selected)
    tiny = select_training(rows, 128)
    negative = [r for r in tiny if r['bucket'] == 'gref_empty']
    assert all(r['native_proposals'] for r in negative)
    rows[0]['generated_query'] = True
    with pytest.raises(ValueError, match='public TRAIN'):
        select_training(rows, 1024)

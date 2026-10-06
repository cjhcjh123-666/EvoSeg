import torch

from projects.evoseg.hallucination.probe_interaction import auc, select_train_cases


def test_probe_labels_are_public_train_and_image_disjoint():
    refs = []
    for index in range(9):
        for empty in (False, True):
            refs.append({'file_name': f'{index}.jpg', 'split': 'train', 'no_target': empty,
                'ann_id': [-1] if empty else [index + 1], 'ref_id': index * 2 + empty,
                'sentences': [{'sent_id': index * 4 + empty * 2 + i, 'sent': f'original public query {i}'}
                              for i in range(2)]})
    refs.append({'file_name': '0.jpg', 'split': 'val'})
    cases = select_train_cases(refs, 8)
    assert len(cases) == 32
    assert all(r['public_split'] == 'train' and r['file_name'] != '0.jpg' for r in cases)
    train = {r['file_name'] for r in cases if not r['probe_holdout']}
    heldout = {r['file_name'] for r in cases if r['probe_holdout']}
    assert not (train & heldout)
    for image in train | heldout:
        selected = [r for r in cases if r['file_name'] == image]
        assert sum(r['private_gt_no_target'] for r in selected) == 2


def test_probe_auc_ties_have_half_credit():
    labels = torch.tensor([True, False, True, False])
    assert auc(labels, torch.tensor([1., 0., 1., 0.])) == 1.
    assert auc(labels, torch.ones(4)) == .5

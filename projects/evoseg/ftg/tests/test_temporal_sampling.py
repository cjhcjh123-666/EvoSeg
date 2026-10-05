import numpy as np
import pytest
import torch
import torch.nn.functional as F

from projects.sa2va.hf.models_qwen3vl_sam3.modeling_sa2va_qwen import (
    resize_mask_logits_to_bool_cpu,
    select_temporal_indices,
)


def test_temporal_indices_preserve_legacy_and_cover_full_video():
    assert select_temporal_indices(12, 5, "first") == [0, 1, 2, 3, 4]
    assert select_temporal_indices(12, 5, "uniform") == [0, 2, 5, 8, 11]
    assert select_temporal_indices(3, 5, "uniform") == [0, 1, 2]
    with pytest.raises(ValueError):
        select_temporal_indices(12, 5, "unknown")


def test_identity_and_state_views_can_be_factorized():
    identity = select_temporal_indices(12, 5, "first")
    state = select_temporal_indices(12, 5, "uniform")
    assert identity == [0, 1, 2, 3, 4]
    assert state == [0, 2, 5, 8, 11]
    assert identity != state


def test_chunked_resize_matches_unbounded_resize_exactly():
    torch.manual_seed(7)
    logits = torch.randn(19, 1, 17, 23)
    expected = (
        F.interpolate(
            logits, size=(31, 29), mode="bilinear", align_corners=False
        )[:, 0].sigmoid() > 0.5
    ).numpy()
    actual = resize_mask_logits_to_bool_cpu(
        logits, size=(31, 29), chunk_size=3
    )
    assert np.array_equal(actual, expected)

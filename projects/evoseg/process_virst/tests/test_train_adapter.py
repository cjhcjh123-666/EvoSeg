import numpy as np

from projects.evoseg.process_virst.long_rvos_train_adapter import ORDER_PATTERN, encode_mask


def test_explicit_order_filter_is_preregistered_and_narrow():
    assert ORDER_PATTERN.search("the man sits down, then picks up the cup")
    assert ORDER_PATTERN.search("the person who runs after waving")
    assert not ORDER_PATTERN.search("the moving person in a blue shirt")


def test_mask_rle_round_trip():
    from pycocotools import mask as mask_utils

    value = np.zeros((8, 9), dtype=bool)
    value[2:6, 3:7] = True
    encoded = encode_mask(value)
    assert encoded is not None
    assert np.array_equal(mask_utils.decode(encoded).astype(bool), value)

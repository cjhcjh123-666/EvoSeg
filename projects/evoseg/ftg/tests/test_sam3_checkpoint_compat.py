from collections import OrderedDict

import torch

from projects.sa2va.models.sa2va import Sa2VAModel


def test_hf_g_weight_is_remapped_to_training_gamma():
    # Avoid constructing Qwen/SAM3: nn.Module initialization is sufficient to
    # exercise Sa2VAModel's checkpoint compatibility hook.
    model = Sa2VAModel.__new__(Sa2VAModel)
    torch.nn.Module.__init__(model)
    model.block = torch.nn.Module()
    model.block.gamma = torch.nn.Parameter(torch.zeros(2))

    incompatible = model.load_state_dict(
        OrderedDict({"block.g_weight": torch.tensor([1.0, 2.0])}),
        strict=True,
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(model.block.gamma, torch.tensor([1.0, 2.0]))


def test_native_key_is_not_rewritten():
    model = Sa2VAModel.__new__(Sa2VAModel)
    torch.nn.Module.__init__(model)
    model.block = torch.nn.Module()
    model.block.g_weight = torch.nn.Parameter(torch.zeros(2))

    model.load_state_dict(
        OrderedDict({"block.g_weight": torch.tensor([3.0, 4.0])}),
        strict=True,
    )

    assert torch.equal(model.block.g_weight, torch.tensor([3.0, 4.0]))

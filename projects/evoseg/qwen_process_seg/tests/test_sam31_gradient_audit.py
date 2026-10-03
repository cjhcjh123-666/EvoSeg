from pathlib import Path

import torch

from projects.evoseg.qwen_process_seg.sam31_gradient_audit import sha256


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "sample"
    path.write_bytes(b"qwen-process-seg")
    assert sha256(path) == "a4d23c7e56c308020154f5dbd3350286f551420020faccb0c7998fae48b5ebd6"


def test_frozen_module_keeps_input_gradient() -> None:
    module = torch.nn.Linear(4, 1)
    module.requires_grad_(False)
    value = torch.randn(2, 4, requires_grad=True)
    module(value).sum().backward()
    assert value.grad is not None
    assert value.grad.norm().item() > 0
    assert all(parameter.grad is None for parameter in module.parameters())

import torch

from projects.evoseg.qwen_process_seg.bridge import (
    FrameQueryFusion,
    QwenToSAM31Bridge,
    multiplex_prompt_batch,
)


def test_frame_prompts_differ() -> None:
    fusion = FrameQueryFusion(8)
    bridge = QwenToSAM31Bridge(8, 4)
    frames = torch.randn(3, 8)
    query = torch.randn(8)
    prompts = bridge(fusion(frames, query))
    assert prompts.shape == (3, 4)
    assert not torch.allclose(prompts[0], prompts[1])


def test_multiplex_prompt_preserves_invalid_slots() -> None:
    prompts = torch.randn(3, 4)
    valid = torch.ones(2, 4)
    invalid = -torch.ones(2, 4)
    result = multiplex_prompt_batch(prompts, valid, invalid)
    assert result.shape == (3, 2, 4)
    assert torch.allclose(result[:, 0], prompts + 1)
    assert torch.all(result[:, 1] == -1)


def test_bridge_gradient() -> None:
    bridge = QwenToSAM31Bridge(8, 4)
    states = torch.randn(2, 8, requires_grad=True)
    bridge(states).square().mean().backward()
    assert states.grad is not None
    assert states.grad.norm().item() > 0

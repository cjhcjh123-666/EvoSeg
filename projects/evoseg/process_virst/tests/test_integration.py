import torch
from torch import nn

from projects.evoseg.process_virst.virst_integration import ProcessAwareSegPrompter, QueryStateCapture
from projects.evoseg.process_virst.process_module import ProcessConditioner


class _Layer(nn.Module):
    def forward(self, tgt, memory, **kwargs):
        batch = tgt.shape[1]
        heads = 2
        attention = torch.ones(batch, heads, tgt.shape[0], memory.shape[0], device=tgt.device)
        attention = attention / memory.shape[0]
        return tgt, attention


class _Official(nn.Module):
    token_dim = 8
    max_length = 100
    grid_hw = 4
    nhead = 2

    def __init__(self):
        super().__init__()
        self.spatial_down = nn.Sequential(
            nn.Conv2d(256, 8, 3, 2, 1), nn.GELU(),
            nn.Conv2d(8, 8, 3, 2, 1), nn.GELU(),
            nn.Conv2d(8, 8, 3, 2, 1), nn.GELU(),
        )
        self.memory_norm = nn.LayerNorm(8)
        self.seg_proj = nn.Linear(16, 8)
        self.decoder = nn.ModuleList([_Layer()])


def test_wrapper_produces_frame_specific_prompts_and_order_diagnostics():
    wrapper = ProcessAwareSegPrompter(_Official(), query_dim=16)
    wrapper.set_query_context(
        torch.randn(1, 5, 16), permutations=("reverse", "block_swap")
    )
    decoded, frame_score = wrapper(
        seg_token=torch.randn(1, 1, 16),
        image_token=torch.randn(1, 6, 256, 32, 32),
        return_attn=True,
    )
    assert decoded.shape == (1, 1, 6, 8)
    assert frame_score.shape == (1, 6)
    assert wrapper.last_diagnostics is not None
    assert wrapper.last_diagnostics.original.alignment.shape == (1, 4, 6)
    assert set(wrapper.last_diagnostics.permuted) == {"reverse", "block_swap"}
    assert wrapper.last_diagnostics.permuted["reverse"].alignment.shape == (1, 4, 6)


def test_videochat_nested_output_hidden_state_is_extracted():
    class Output:
        last_hidden_state = torch.randn(1, 3, 4)

    expected = Output.last_hidden_state
    assert QueryStateCapture._last_hidden_state((Output(), torch.ones(1, 3))) is expected


def test_global_alignment_control_is_frame_permutation_invariant():
    torch.manual_seed(3)
    module = ProcessConditioner(
        query_dim=16,
        vision_dim=256,
        process_dim=8,
        slots=4,
        heads=2,
        alignment_mode="global",
    )
    query = torch.randn(1, 5, 16)
    video = torch.randn(1, 6, 256, 32, 32)
    original = module(query, video).alignment_score
    reversed_score = module(query, video.flip(1)).alignment_score
    assert torch.allclose(original, reversed_score, atol=1e-5)

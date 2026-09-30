import torch
from torch import nn

from projects.evoseg.process_virst.virst_integration import ProcessAwareSegPrompter


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
    wrapper.set_query_context(torch.randn(1, 5, 16), permutation="reverse")
    decoded, frame_score = wrapper(
        seg_token=torch.randn(1, 1, 16),
        image_token=torch.randn(1, 6, 256, 32, 32),
        return_attn=True,
    )
    assert decoded.shape == (1, 1, 6, 8)
    assert frame_score.shape == (1, 6)
    assert wrapper.last_diagnostics is not None
    assert wrapper.last_diagnostics.original.alignment.shape == (1, 4, 6)
    assert wrapper.last_diagnostics.permuted_score is not None

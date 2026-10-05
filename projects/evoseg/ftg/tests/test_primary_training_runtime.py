import torch

from projects.evoseg.ftg.primary_model import EvidenceFactorization
from projects.evoseg.ftg.train_primary import select_development_indices


def test_development_is_balanced_and_uses_distinct_videos():
    records = [{"dataset": source, "expression_type": kind, "video_id": str(video)}
               for source in ("long", "mevis") for video in range(9)
               for kind in ("static", "dynamic", "hybrid")]
    selected = select_development_indices(records, 12)
    assert selected == select_development_indices(records, 12)
    assert len(selected) == 12
    assert len({(records[i]["dataset"], records[i]["video_id"]) for i in selected}) == 12
    assert sum(records[i]["dataset"] == "long" for i in selected) == 6
    assert {records[i]["expression_type"] for i in selected} == {"static", "dynamic", "hybrid"}


def test_no_grad_autocast_evaluation_then_training_keeps_all_gradients():
    module = EvidenceFactorization(dim=16, heads=4)
    identity, semantics, spatial = torch.randn(16), torch.randn(4, 16), torch.randn(4, 9, 16)
    module.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        module(identity, semantics, spatial)
    module.train()
    torch.clear_autocast_cache()
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        prompts, _ = module(identity, semantics, spatial)
        loss = prompts.float().square().mean()
    loss.backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

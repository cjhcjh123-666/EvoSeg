import torch

from projects.evoseg.ftg.interface import FTG_VARIANTS, FactorizedTemporalGrounding


def test_all_controlled_variants_produce_valid_prompts():
    torch.manual_seed(3)
    frames = torch.randn(5, 12)
    query = torch.randn(12)
    for variant in FTG_VARIANTS:
        module = FactorizedTemporalGrounding(
            hidden_dim=12, prompt_dim=7, bottleneck_dim=9, variant=variant
        )
        prompts, diagnostics = module(frames, query)
        assert prompts.shape == (5, 7)
        assert torch.isfinite(prompts).all()
        assert diagnostics["grounding_variant"] == variant
        assert 0 <= diagnostics["gate_mean"].item() <= 1
        active = module.activate_variant_parameters()
        assert active
        assert sum(parameter.numel() for parameter in active) == module.active_parameter_count()


def test_persistent_controls_are_constant_and_temporal_controls_vary():
    torch.manual_seed(7)
    frames = torch.randn(4, 10)
    query = torch.randn(10)
    for variant in ("global_prompt", "identity_only"):
        module = FactorizedTemporalGrounding(10, 6, 8, variant)
        prompts, _ = module(frames, query)
        assert torch.equal(prompts, prompts[:1].expand_as(prompts))
    for variant in ("frame_prompt", "state_only", "id_state_no_gate", "ftg"):
        module = FactorizedTemporalGrounding(10, 6, 8, variant)
        prompts, _ = module(frames, query)
        assert prompts.float().std(dim=0).mean() > 0


def test_ftg_is_frame_permutation_equivariant_with_invariant_identity():
    torch.manual_seed(13)
    module = FactorizedTemporalGrounding(11, 5, 8, "ftg")
    frames = torch.randn(6, 11)
    query = torch.randn(11)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    original, original_diagnostics = module(frames, query)
    shuffled, shuffled_diagnostics = module(frames[permutation], query)
    torch.testing.assert_close(
        original_diagnostics["identity_token"], shuffled_diagnostics["identity_token"]
    )
    torch.testing.assert_close(original[permutation], shuffled)

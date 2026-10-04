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
    for variant in ("frame_prompt", "state_only"):
        module = FactorizedTemporalGrounding(10, 6, 8, variant)
        prompts, _ = module(frames, query)
        assert prompts.float().std(dim=0).mean() > 0


def test_factorized_residual_starts_at_identity_only_prompt():
    torch.manual_seed(9)
    module = FactorizedTemporalGrounding(10, 6, 8, "identity_only")
    frames = torch.randn(4, 10)
    query = torch.randn(10)
    identity_prompts, _ = module(frames, query)
    module.variant = "id_state_no_gate"
    ungated_prompts, _ = module(frames, query)
    module.variant = "ftg"
    gated_prompts, _ = module(frames, query)
    torch.testing.assert_close(identity_prompts, ungated_prompts)
    torch.testing.assert_close(identity_prompts, gated_prompts)


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


def test_ftg_association_is_persistent_but_frame_control_is_not():
    torch.manual_seed(15)
    frames = torch.randn(5, 11)
    query = torch.randn(11)
    ftg = FactorizedTemporalGrounding(11, 5, 8, "ftg")
    frame = FactorizedTemporalGrounding(11, 5, 8, "frame_prompt")

    _, ftg_diagnostics = ftg(frames, query)
    _, frame_diagnostics = frame(frames, query)

    ftg_association = ftg_diagnostics["association_prompts"]
    frame_association = frame_diagnostics["association_prompts"]
    assert torch.equal(ftg_association, ftg_association[:1].expand_as(ftg_association))
    assert frame_association.float().std(dim=0).mean() > 0


def test_native_anchor_active_residual_can_start_exactly_zero():
    torch.manual_seed(17)
    frames = torch.randn(4, 10)
    query = torch.randn(10)
    for variant in FTG_VARIANTS:
        module = FactorizedTemporalGrounding(10, 6, 8, variant)
        module.zero_active_output_projection()
        prompts, diagnostics = module(frames, query)
        if variant == "ftg":
            prompts = diagnostics["state_gate"] * diagnostics["state_prompts"]
        elif variant == "id_state_no_gate":
            prompts = diagnostics["state_prompts"]
        elif variant == "identity_only":
            prompts = diagnostics["identity_prompts"]
        elif variant == "state_only":
            prompts = diagnostics["independent_state_prompts"]
        assert torch.count_nonzero(prompts) == 0

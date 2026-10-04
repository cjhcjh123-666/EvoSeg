from projects.evoseg.ftg.convert_strong_hf_to_pth import _destination_key


def test_qwen_and_projection_keys_are_remapped_for_training_model():
    assert _destination_key("model.model.layers.0.weight", "all") == (
        "mllm.model.model.layers.0.weight"
    )
    assert _destination_key("text_hidden_fcs.0.weight", "all") == (
        "text_hidden_fcs.0.weight"
    )


def test_sam_scope_can_keep_all_or_only_decoder():
    backbone = "grounding_encoder.sam2_model.backbone.weight"
    decoder = "grounding_encoder.sam2_model.sam_mask_decoder.weight"
    assert _destination_key(backbone, "all") == backbone
    assert _destination_key(backbone, "decoder") is None
    assert _destination_key(decoder, "decoder") == decoder


def test_unrelated_export_keys_are_ignored():
    assert _destination_key("unused.weight", "all") is None

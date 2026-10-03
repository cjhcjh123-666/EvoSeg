import json

from projects.evoseg.ftg.summarize_pilot import summarize


GROUPS = (
    "long_rvos/static",
    "long_rvos/dynamic",
    "long_rvos/hybrid",
    "mevis_v2/motion",
)


def _write_result(root, variant, values):
    directory = root / variant
    directory.mkdir()
    groups = {
        group: {"jf": value, "count": 2 if group.startswith("long") else 6}
        for group, value in zip(GROUPS, values, strict=True)
    }
    payload = {
        "variant": variant,
        "manifest_sha256": "same",
        "validation": {"original": groups},
    }
    (directory / "result.json").write_text(json.dumps(payload))


def test_gate_can_pass_while_claim_remains_mixed(tmp_path):
    _write_result(tmp_path, "frame_prompt", (0.10, 0.10, 0.10, 0.10))
    _write_result(tmp_path, "ftg", (0.11, 0.10, 0.20, 0.07))
    result = summarize(tmp_path, selective_gain=0.01, static_floor=-0.005)
    assert result["decision"] == "GO"
    assert result["claim_status"] == "MIXED"
    assert result["direction_consistent"] is False
    assert result["small_group_warning"] is True


def test_consistent_selective_gain_supports_claim(tmp_path):
    _write_result(tmp_path, "frame_prompt", (0.10, 0.10, 0.10, 0.10))
    _write_result(tmp_path, "ftg", (0.101, 0.12, 0.13, 0.14))
    result = summarize(tmp_path, selective_gain=0.01, static_floor=-0.005)
    assert result["decision"] == "GO"
    assert result["claim_status"] == "SUPPORTED"

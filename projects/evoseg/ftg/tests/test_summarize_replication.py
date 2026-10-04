import json

from projects.evoseg.ftg.summarize_replication import summarize_pairs


GROUPS = (
    "long_rvos/static", "long_rvos/dynamic", "long_rvos/hybrid",
    "mevis_v2", "overall",
)


def _result(path, variant, values, manifest="same"):
    groups = {
        group: {
            "jf": value,
            "count": 6,
            "oracle_jf": value + 0.1,
            "query_selection_accuracy": 0.5,
            "predicted_query_switch_rate": 0.2,
        }
        for group, value in zip(GROUPS, values, strict=True)
    }
    path.write_text(json.dumps({
        "status": "COMPLETE",
        "variant": variant,
        "manifest_sha256": manifest,
        "validation": {"original": groups},
    }))


def test_replication_reports_mean_std_and_sign_consistency(tmp_path):
    pairs = []
    for seed, ftg_shift in ((11, 0.02), (23, 0.04)):
        frame = tmp_path / f"frame_{seed}.json"
        ftg = tmp_path / f"ftg_{seed}.json"
        _result(frame, "frame_prompt", (0.2,) * 5)
        _result(ftg, "ftg", (0.201, 0.2 + ftg_shift, 0.23, 0.24, 0.22))
        pairs.append((f"seed{seed}", frame, ftg))

    result = summarize_pairs(pairs)

    assert result["pair_count"] == 2
    assert result["aggregate"]["long_rvos/hybrid"]["positive_pairs"] == 2
    assert result["aggregate"]["long_rvos/dynamic"]["delta"]["mean"] == 0.03
    assert result["decision"] == "GO"


def test_replication_rejects_manifest_mismatch(tmp_path):
    frame = tmp_path / "frame.json"
    ftg = tmp_path / "ftg.json"
    _result(frame, "frame_prompt", (0.2,) * 5, manifest="a")
    _result(ftg, "ftg", (0.3,) * 5, manifest="b")

    try:
        summarize_pairs([("bad", frame, ftg)])
    except RuntimeError as error:
        assert "different manifests" in str(error)
    else:
        raise AssertionError("manifest mismatch should fail")

import json
from pathlib import Path

from projects.evoseg.qwen_process_seg.long_rvos import build_capability_manifest


def test_manifest_uses_official_expression_and_full_range(tmp_path: Path) -> None:
    root = tmp_path
    (root / "JPEGImages" / "v").mkdir(parents=True)
    (root / "Annotations" / "v" / "1").mkdir(parents=True)
    frames = [f"{index:05d}" for index in range(20)]
    for name in frames:
        (root / "JPEGImages" / "v" / f"{name}.jpg").touch()
        (root / "Annotations" / "v" / "1" / f"{name}.png").touch()
    (root / "meta_expressions.json").write_text(json.dumps({"videos": {"v": {
        "frames": frames,
        "expressions": {"e": {"exp": "official query", "obj_id": 1, "type": "dynamic"}},
    }}}))
    record = build_capability_manifest(root, count=1)["records"][0]
    assert record["expression"] == "official query"
    assert record["source_frame_indices"][0] == 0
    assert record["source_frame_indices"][-1] == 19

import json

from projects.evoseg.process_virst.compare_long_rvos import compare


def test_long_comparison_is_expression_matched_and_object_balanced(tmp_path):
    manifest = {
        "objects": [
            {
                "dataset": "d",
                "video_id": "v",
                "object_id": 1,
                "expressions": [
                    {"expression_id": "s", "type": "static", "text": "the person"},
                    {"expression_id": "d", "type": "dynamic", "text": "the person then sits"},
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    baseline_path = tmp_path / "baseline.jsonl"
    process_path = tmp_path / "process.jsonl"
    base_rows = []
    process_rows = []
    for expression_id, value in (("s", 0.5), ("d", 0.4)):
        key = f"d/v/1/{expression_id}/native"
        base_rows.append({"key": key, "status": "success", "J": value, "F": value, "J_and_F": value})
        process_rows.append({"key": key, "status": "success", "J": value + 0.1, "F": value + 0.1, "J_and_F": value + 0.1})
    baseline_path.write_text("".join(json.dumps(row) + "\n" for row in base_rows))
    process_path.write_text("".join(json.dumps(row) + "\n" for row in process_rows))
    result = compare(
        manifest_path,
        [baseline_path],
        [process_path],
        tmp_path / "rows.csv",
        tmp_path / "summary.json",
        iterations=10,
        seed=42,
    )
    assert abs(result["groups"]["dynamic"]["delta_J_and_F"] - 0.1) < 1e-9
    assert result["groups"]["explicit_order"]["expressions"] == 1

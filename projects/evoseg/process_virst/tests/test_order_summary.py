import json

from projects.evoseg.process_virst.summarize_order_diagnostics import summarize


def test_order_summary_reports_original_minus_permutation(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "alignment_score": [2.0],
                "beta": 1.0,
                "permutations": {
                    "reverse": {
                        "alignment_score": [1.5],
                        "frame_state_response_l2": [[0.2, 0.4]],
                    }
                },
            }
        )
        + "\n"
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps([{"video_id": "v", "expression_id": "e", "question": "q"}])
    )
    result = summarize(
        diagnostics,
        mapping,
        tmp_path / "rows.csv",
        tmp_path / "summary.json",
        iterations=10,
        seed=42,
    )
    assert result["reverse"]["mean"] == 0.5

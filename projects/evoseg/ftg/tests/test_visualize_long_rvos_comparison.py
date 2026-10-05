from projects.evoseg.ftg.visualize_long_rvos_comparison import (
    paired_records,
    selected_extremes,
)


def _row(kind, j, f):
    return {"type": kind, "j": j, "f": f}


def test_paired_records_rank_official_jf_delta_by_type():
    baseline = {
        ("v", "0"): _row("static", 0.4, 0.6),
        ("v", "1"): _row("static", 0.8, 0.8),
        ("v", "2"): _row("dynamic", 0.2, 0.4),
        ("v", "3"): _row("dynamic", 0.6, 0.6),
        ("v", "4"): _row("hybrid", 0.3, 0.5),
        ("v", "5"): _row("hybrid", 0.7, 0.7),
    }
    candidate = {
        ("v", "0"): _row("static", 0.5, 0.7),
        ("v", "1"): _row("static", 0.6, 0.6),
        ("v", "2"): _row("dynamic", 0.5, 0.5),
        ("v", "3"): _row("dynamic", 0.5, 0.5),
        ("v", "4"): _row("hybrid", 0.6, 0.6),
        ("v", "5"): _row("hybrid", 0.4, 0.6),
    }
    records = paired_records(baseline, candidate)
    selected = selected_extremes(records, per_side=1)
    assert len(selected) == 6
    for kind in ("static", "dynamic", "hybrid"):
        typed = [(side, row) for side, row in selected if row["type"] == kind]
        assert [side for side, _ in typed] == ["worst", "best"]
        assert typed[0][1]["delta_jf"] <= typed[1][1]["delta_jf"]

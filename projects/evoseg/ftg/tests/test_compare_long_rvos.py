import pytest

from projects.evoseg.eval.compare_long_rvos import (
    cluster_bootstrap_mean,
    paired_deltas,
    summarize,
)


def _row(kind, j, f):
    return {"type": kind, "j": j, "f": f, "tiou": j, "viou": f}


def test_paired_summary_and_video_cluster_bootstrap_are_reproducible():
    baseline = {
        ("v1", "0"): _row("dynamic", 0.4, 0.6),
        ("v1", "1"): _row("dynamic", 0.3, 0.5),
        ("v2", "0"): _row("static", 0.8, 0.8),
    }
    candidate = {
        ("v1", "0"): _row("dynamic", 0.5, 0.7),
        ("v1", "1"): _row("dynamic", 0.2, 0.4),
        ("v2", "0"): _row("static", 0.9, 0.9),
    }
    values = paired_deltas(baseline, candidate, metric="j_and_f")
    summary = summarize(values, bootstrap_samples=200, seed=7)
    assert summary["n"] == 3
    assert summary["videos"] == 2
    assert summary["mean_delta"] == pytest.approx(1 / 30)
    assert summary["positive_fraction"] == pytest.approx(2 / 3)
    assert cluster_bootstrap_mean(values, 200, 7) == cluster_bootstrap_mean(
        values, 200, 7
    )


def test_type_filter_keeps_only_requested_expressions():
    baseline = {
        ("v1", "0"): _row("dynamic", 0.4, 0.6),
        ("v2", "0"): _row("static", 0.8, 0.8),
    }
    candidate = {
        ("v1", "0"): _row("dynamic", 0.5, 0.7),
        ("v2", "0"): _row("static", 0.9, 0.9),
    }
    values = paired_deltas(
        baseline, candidate, expression_type="dynamic", metric="j_and_f"
    )
    assert len(values) == 1
    assert values[0]["video"] == "v1"

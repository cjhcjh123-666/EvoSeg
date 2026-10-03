"""Summarize the preregistered FTG-vs-frame-prompt pilot decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(result: dict, group: str) -> float:
    return result["validation"]["original"][group]["jf"]


def summarize(root: Path, selective_gain: float, static_floor: float) -> dict:
    results = {}
    for directory in sorted(root.iterdir()):
        path = directory / "result.json"
        if path.is_file():
            result = json.loads(path.read_text())
            results[result["variant"]] = result
    required = {"frame_prompt", "ftg"}
    if not required.issubset(results):
        raise RuntimeError(f"missing completed variants: {sorted(required - results.keys())}")
    frame = results["frame_prompt"]
    ftg = results["ftg"]
    if frame["manifest_sha256"] != ftg["manifest_sha256"]:
        raise RuntimeError("controlled variants used different manifests")

    groups = (
        "long_rvos/static", "long_rvos/dynamic", "long_rvos/hybrid", "mevis_v2/motion"
    )
    deltas = {group: _metric(ftg, group) - _metric(frame, group) for group in groups}
    dynamic_mean = (
        deltas["long_rvos/dynamic"]
        + deltas["long_rvos/hybrid"]
        + deltas["mevis_v2/motion"]
    ) / 3
    passed = dynamic_mean >= selective_gain and deltas["long_rvos/static"] >= static_floor
    summary = {
        "decision": "GO" if passed else "NO-GO",
        "comparison": "ftg_minus_frame_prompt",
        "manifest_sha256": ftg["manifest_sha256"],
        "thresholds": {
            "dynamic_mean_minimum": selective_gain,
            "static_minimum": static_floor,
        },
        "deltas": deltas,
        "dynamic_motion_mean_delta": dynamic_mean,
        "available_variants": sorted(results),
    }
    (root / "controlled_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--selective-gain", type=float, default=0.01)
    parser.add_argument("--static-floor", type=float, default=-0.005)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(summarize(arguments.root, arguments.selective_gain, arguments.static_floor), indent=2))

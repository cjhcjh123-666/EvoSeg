"""Merge deterministic Qwen-seeded tracker evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.evoseg.ftg.evaluate_qwen_seeded_tracker import _aggregate, _atomic_json


def run(inputs: list[Path], output: Path, expected_count: int | None) -> dict:
    payloads = [json.loads((path / "result.json").read_text()) for path in inputs]
    records = [record for payload in payloads for record in payload["records"]]
    keys = [
        (record["dataset"], record["video_id"], record["expression_id"])
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate expressions found across evaluation shards")
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(f"expected {expected_count} records, found {len(records)}")
    result = {
        "status": "COMPLETE",
        "method": "Qwen Frame Prompt mask -> official SAM3.1 tracker memory",
        "ground_truth_used_for_initialization_or_selection": False,
        "shard_count": len(payloads),
        "record_count": len(records),
        "frame_prompt": _aggregate(records, "frame"),
        "qwen_seeded_tracker": _aggregate(records, "tracker"),
        "peak_memory_gib_per_worker": max(
            payload["peak_memory_gib"] for payload in payloads
        ),
        "records": sorted(
            records,
            key=lambda record: (
                record["dataset"], record["video_id"], record["expression_id"]
            ),
        ),
    }
    _atomic_json(output / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run(arguments.inputs, arguments.output, arguments.expected_count)
    print(json.dumps(summary["qwen_seeded_tracker"], indent=2))

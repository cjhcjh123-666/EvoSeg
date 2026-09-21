"""Validate the two-video smoke run before expanding the queue."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SUCCESS = {"success", "success_no_seg"}


def load_latest(path):
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["key"]] = record
    return list(records.values())


def check(run_dir: Path, expected_videos=2):
    records = load_latest(run_dir / "predictions.jsonl")
    smoke = [r for r in records if r.get("phase_first_seen") == "smoke"]
    errors = []
    warnings = []
    if len({r["key"] for r in smoke}) != len(smoke):
        errors.append("duplicate result keys")
    videos = {r["video_id"] for r in smoke}
    if len(videos) != expected_videos:
        errors.append(f"expected {expected_videos} videos, found {len(videos)}")
    unexpected_failures = [r for r in smoke if r.get("status") == "failed"]
    if unexpected_failures:
        errors.append(f"{len(unexpected_failures)} non-OOM inference failures")
    oom = [r for r in smoke if r.get("status") == "failed_oom"]
    if oom:
        warnings.append(f"{len(oom)} OOM conditions retained as failures")

    groups = defaultdict(list)
    for record in smoke:
        groups[(record["video_id"], record["object_id"], record["expression_id"])].append(record)
        if record.get("gt_entered_model") is not False:
            errors.append(f"GT information-flow assertion failed: {record['key']}")
        if record.get("status") in SUCCESS:
            if len(record["segmentation_frame_names"]) != len(record["evaluation_frame_names"]):
                errors.append(f"output/evaluation length mismatch: {record['key']}")
            if not (run_dir / record["rle_masks_path"]).is_file():
                errors.append(f"missing RLE file: {record['key']}")
    for key, values in groups.items():
        signatures = {v["segmentation_frame_signature"] for v in values}
        prompt_positions = {tuple(v["sam2_prompt_positions"]) for v in values}
        if len(signatures) != 1 or len(prompt_positions) != 1:
            errors.append(f"segmentation/prompt frames changed across budgets: {key}")
        successes = sorted(
            (int(v["frame_budget"]), v["token_info"]["visual_tokens"])
            for v in values
            if v.get("status") in SUCCESS
        )
        if len(successes) >= 2:
            if any(b <= a for (_n1, a), (_n2, b) in zip(successes, successes[1:])):
                errors.append(f"visual token count did not increase: {key}: {successes}")

    config = json.loads((run_dir / "run_config.json").read_text())
    if not config["runtime_audit"]["gate_audit_passed"]:
        errors.append("runtime temporal-gate audit failed")
    successful = [r for r in smoke if r.get("status") in SUCCESS]
    if not successful:
        errors.append("no real inference succeeded")
    result = {
        "passed": not errors,
        "videos": len(videos),
        "results": len(smoke),
        "successful": len(successful),
        "oom": len(oom),
        "errors": errors,
        "warnings": warnings,
    }
    (run_dir / "logs" / "smoke_quality.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-videos", type=int, default=2)
    args = parser.parse_args()
    result = check(Path(args.run_dir).resolve(), args.expected_videos)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()


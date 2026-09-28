"""Verify reconstructed frozen probe checkpoints reproduce the preregistered pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args) -> None:
    old_path = Path(args.original)
    new_path = Path(args.reconstructed)
    old = list(csv.DictReader(old_path.open()))
    new = list(csv.DictReader(new_path.open()))
    if len(old) != len(new):
        raise RuntimeError(f"row count differs: {len(old)} != {len(new)}")
    columns = [
        "identity", "candidate_hit", "oracle_track_id",
        "static_selected_track_id", "static_selection_correct", "static_candidate_direct_J_and_F", "static_target_margin",
        "temporal_selected_track_id", "temporal_selection_correct", "temporal_candidate_direct_J_and_F", "temporal_target_margin",
    ]
    mismatch = {column: 0 for column in columns}
    max_float_delta = {column: 0.0 for column in columns if "J_and_F" in column or "margin" in column}
    for left, right in zip(old, new):
        for column in columns:
            if column in max_float_delta and left[column] and right[column]:
                delta = abs(float(left[column]) - float(right[column]))
                max_float_delta[column] = max(max_float_delta[column], delta)
                mismatch[column] += int(delta != 0.0)
            else:
                mismatch[column] += int(left[column] != right[column])
    audit = {
        "original": str(old_path.resolve()), "reconstructed": str(new_path.resolve()),
        "original_sha256": sha256(old_path), "reconstructed_sha256": sha256(new_path),
        "rows": len(old), "columns_checked": columns, "mismatch_counts": mismatch,
        "max_float_delta": max_float_delta,
        "exact_metric_and_selection_reproduction": not any(mismatch.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["exact_metric_and_selection_reproduction"]:
        raise RuntimeError(f"probe reconstruction differs: {audit}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--reconstructed", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

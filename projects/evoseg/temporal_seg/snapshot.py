"""Materialize an immutable, reproducible first-round experiment snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .summarize import summarize


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def materialize(run_dir: Path, cutoff: str, name: str = "first_round_8h"):
    run_dir = run_dir.resolve()
    cutoff_time = parse_time(cutoff)
    destination = run_dir / "snapshots" / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "logs").mkdir(exist_ok=True)

    source_predictions = run_dir / "predictions.jsonl"
    selected = []
    for line in source_predictions.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        completed = record.get("completed_at")
        if completed and parse_time(completed) <= cutoff_time:
            selected.append(record)
    snapshot_predictions = destination / "predictions.jsonl"
    with snapshot_predictions.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for filename in ("manifest.json", "run_config.json", "LITERATURE_MATRIX.md"):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, destination / filename)
    quality = run_dir / "logs" / "smoke_quality.json"
    if quality.exists():
        shutil.copy2(quality, destination / "logs" / quality.name)

    masks_link = destination / "masks"
    if not masks_link.exists() and not masks_link.is_symlink():
        os.symlink("../../masks", masks_link)
    (destination / "STATUS.json").write_text(
        json.dumps(
            {
                "state": "first_round_snapshot",
                "snapshot_name": name,
                "snapshot_cutoff": cutoff,
                "source_run_dir": str(run_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    summary_status = summarize(destination)
    metadata = {
        "schema_version": 1,
        "name": name,
        "cutoff": cutoff,
        "source_run_dir": str(run_dir),
        "prediction_records": len(selected),
        "successful_results": summary_status.get("successful_results", 0),
        "failed_results": summary_status.get("failed_results", 0),
        "covered_videos": summary_status.get("covered_videos", 0),
        "covered_objects": summary_status.get("covered_objects", 0),
        "predictions_sha256": sha256(snapshot_predictions),
    }
    (destination / "SNAPSHOT.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--name", default="first_round_8h")
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(Path(args.run_dir), args.cutoff, args.name),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

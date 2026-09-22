"""Merge independently written representation shards with strict key audits."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_records(paths: list[Path]) -> tuple[list[dict], dict]:
    merged: dict[str, dict] = {}
    duplicates: list[dict] = []
    for path in paths:
        for record in load_jsonl(path):
            key = record["key"]
            if key in merged:
                previous = merged[key]
                same_result = (
                    previous.get("status") == record.get("status")
                    and previous.get("vector_sha256") == record.get("vector_sha256")
                )
                duplicates.append(
                    {
                        "key": key,
                        "path": str(path),
                        "same_result": same_result,
                    }
                )
                if not same_result:
                    raise RuntimeError(
                        f"conflicting duplicate representation key {key!r} in {path}"
                    )
                continue
            merged[key] = record
    rows = list(merged.values())
    audit = {
        "input_paths": [str(path) for path in paths],
        "input_rows": sum(len(load_jsonl(path)) for path in paths),
        "unique_rows": len(rows),
        "success_rows": sum(row.get("status") == "success" for row in rows),
        "failed_rows": sum(row.get("status") != "success" for row in rows),
        "duplicate_rows": len(duplicates),
        "duplicates": duplicates,
    }
    return rows, audit


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    output = run_dir / "representation_records.jsonl"
    worker_paths = sorted(run_dir.glob("representation_records.worker-*-of-*.jsonl"))
    paths = [output, *worker_paths]
    rows, audit = merge_records(paths)
    audit.update(
        {
            "merged_at": utc_now(),
            "expected_count": args.expected_count,
            "complete": (
                audit["success_rows"] == args.expected_count
                and audit["failed_rows"] == 0
            ),
        }
    )
    if not audit["complete"]:
        atomic_json(run_dir / "representation_merge_audit.json", audit)
        raise RuntimeError(
            "representation shards are incomplete: "
            f"success={audit['success_rows']}, failed={audit['failed_rows']}, "
            f"expected={args.expected_count}"
        )
    temporary = output.with_suffix(output.suffix + ".merged.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    atomic_json(run_dir / "representation_merge_audit.json", audit)
    status = {
        "state": "complete",
        "phase": "full",
        "updated_at": utc_now(),
        "completed": audit["success_rows"],
        "planned": args.expected_count,
        "failed": audit["failed_rows"],
        "worker_status_files": [
            path.name.replace("representation_records.", "STATUS.").replace(
                ".jsonl", ".json"
            )
            for path in worker_paths
        ],
        "merge_audit": "representation_merge_audit.json",
    }
    atomic_json(run_dir / "STATUS.json", status)
    print(json.dumps(audit, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=3567)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

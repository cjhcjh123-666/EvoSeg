"""Merge disjoint stage-state shards without copying large NPZ files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args) -> None:
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    statuses = []
    failed_records = {}
    for root in inputs:
        status = json.loads((root / "STAGE_STATUS.json").read_text())
        statuses.append(status)
        accepted_states = {"complete", "complete_with_failures"} if args.allow_failures else {"complete"}
        if status["state"] not in accepted_states or (
            not args.allow_failures and status["failed_this_process"] != 0
        ):
            raise RuntimeError(f"stage shard incomplete: {root}: {status}")
        with (root / "stage_grounding_records.jsonl").open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if value.get("status") != "success":
                    if not args.allow_failures:
                        raise RuntimeError(f"unsuccessful shard record: {value['identity']}")
                    failed_records[value["identity"]] = {
                        "identity": value["identity"],
                        "status": value.get("status"),
                        "error": value.get("error"),
                        "source": str(root),
                    }
                    continue
                if value["identity"] in records:
                    raise RuntimeError(f"duplicate identity: {value['identity']}")
                value = dict(value)
                value["state_path"] = str((root / value["state_path"]).resolve())
                records[value["identity"]] = value
    missing = sorted(set(failed_records) - set(records))
    path = output / "stage_grounding_records.jsonl"
    with path.open("w") as handle:
        for identity in sorted(records):
            handle.write(json.dumps(records[identity], ensure_ascii=False) + "\n")
    atomic_json(
        output / "stage_merge_audit.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": [str(value) for value in inputs],
            "input_statuses": statuses,
            "successful_unique_expressions": len(records),
            "failed_attempts": len(failed_records),
            "missing_identities": missing,
            "missing_identity_count": len(missing),
            "allow_failures": args.allow_failures,
            "duplicate_identities": 0,
            "large_state_files_copied": False,
            "state_paths_are_absolute_to_immutable_shards": True,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

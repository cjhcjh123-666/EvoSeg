"""Seed a new object-level shard layout from successful interrupted caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    object_index = {
        (str(item["video_id"]), str(item["object_id"])): index
        for index, item in enumerate(manifest["objects"][: args.max_objects])
    }
    successful = {}
    source_by_identity = {}
    for source in map(lambda value: Path(value).resolve(), args.input):
        path = source / "stage_grounding_records.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("status") != "success":
                continue
            identity = value["identity"]
            if identity in successful:
                raise RuntimeError(f"duplicate successful identity: {identity}")
            value = dict(value)
            value["state_path"] = str((source / value["state_path"]).resolve())
            successful[identity] = value
            source_by_identity[identity] = str(source)

    output_prefix = Path(args.output_prefix).resolve()
    counts = []
    for shard in range(args.num_shards):
        output = Path(f"{output_prefix}{shard}")
        output.mkdir(parents=True, exist_ok=True)
        selected = []
        for identity, value in successful.items():
            index = object_index[(str(value["video_id"]), str(value["object_id"]))]
            if index % args.num_shards == shard:
                selected.append(value)
        with (output / "stage_grounding_records.jsonl").open("w") as handle:
            for value in sorted(selected, key=lambda row: row["identity"]):
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        counts.append(len(selected))
    audit = {
        "inputs": [str(Path(value).resolve()) for value in args.input],
        "successful_seed_records": len(successful),
        "num_shards": args.num_shards,
        "seed_records_per_shard": counts,
        "source_by_identity": source_by_identity,
        "prediction_or_metric_used_for_resharding": False,
        "assignment": "original prediction-independent object index modulo num_shards",
    }
    audit_path = Path(f"{output_prefix}_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


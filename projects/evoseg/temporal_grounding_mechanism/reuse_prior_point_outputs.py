"""Reuse prior TGI tracking only when the public SAM prompt plan is identical."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from projects.evoseg.temporal_grounding_interface.run_dynamic_interface import (
    tracking_signature,
)


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args) -> None:
    prior = {}
    for path_value in glob.glob(str(Path(args.prior_root) / "**/dynamic_predictions.jsonl"), recursive=True):
        path = Path(path_value)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success" and row.get("sam31_tracking_signature"):
                    prior[(row["identity"], row["sam31_tracking_signature"])] = (path, row)
    manifest = json.loads(Path(args.manifest).read_text())
    object_position = {
        (item["video_id"], str(item["object_id"])): index
        for index, item in enumerate(manifest["objects"][: args.max_objects])
    }
    output_root = Path(args.output_root)
    done = set()
    for path in output_root.glob("point_shard_*/point_results.jsonl"):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "success":
                    done.add((row["identity"], row["condition"]))
    reused = 0
    with Path(args.plans).open() as handle:
        for line in handle:
            plan = json.loads(line)
            result_key = (plan["identity"], plan["condition"])
            signature = tracking_signature(plan["video_id"], plan["plan"])
            source = prior.get((plan["identity"], signature))
            if source is None or result_key in done:
                continue
            source_path, prior_row = source
            shard = object_position[(plan["video_id"], str(plan["object_id"]))] % args.num_shards
            row = {
                key: plan[key]
                for key in (
                    "identity",
                    "dataset",
                    "video_id",
                    "object_id",
                    "expression_id",
                    "description_type",
                    "expression",
                    "condition",
                    "ORACLE",
                )
            }
            row.update(
                status="success",
                J=prior_row["J"],
                F=prior_row["F"],
                J_and_F=prior_row["J_and_F"],
                selected_track_ids=[
                    value["selected_candidate_object_id"] for value in plan["plan"]
                ],
                prompt_updates=sum(value["update_applied"] for value in plan["plan"]),
                latency_seconds_synchronized=prior_row[
                    "sam31_latency_seconds_synchronized"
                ],
                peak_memory_bytes=prior_row["component_peak_memory_bytes"][
                    "sam31_tracking"
                ],
                session_api_compat=prior_row["session_api_compat"],
                gt_entered_model=False,
                cache_hit=True,
                reused_prior_tgi=True,
                reused_tracking_signature=signature,
                reused_source_condition=prior_row["condition"],
                reused_source_file=str(source_path),
            )
            append(output_root / f"point_shard_{shard}" / "point_results.jsonl", row)
            done.add(result_key)
            reused += 1
    print(json.dumps({"exact_signature_matches_reused": reused, "successful_rows": len(done)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

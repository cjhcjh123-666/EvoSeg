"""Launch deterministic Qwen-seeded tracker evaluation shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty run directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    launches = []
    for shard_index, physical_gpu in enumerate(args.gpus):
        run_dir = args.output / f"shard_{shard_index:02d}"
        run_dir.mkdir()
        command = [
            sys.executable,
            "-m", "projects.evoseg.ftg.evaluate_qwen_seeded_tracker",
            "--checkpoint", str(args.checkpoint),
            "--manifest", str(args.manifest),
            "--output", str(run_dir),
            "--device", "0",
            "--shard-index", str(shard_index),
            "--num-shards", str(len(args.gpus)),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        log_path = run_dir / "console.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launches.append(
            {
                "shard_index": shard_index,
                "physical_gpu": physical_gpu,
                "pid": process.pid,
                "output": str(run_dir),
                "log": str(log_path),
                "command": command,
            }
        )
    state = {
        "status": "LAUNCHED",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "launches": launches,
    }
    (args.output / "launch_state.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

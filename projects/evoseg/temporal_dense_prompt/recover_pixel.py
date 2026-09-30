"""Retry an existing TDSP pixel run with one SAM3.1 worker per GPU.

This recovery keeps the original eight-way shard assignment so successful
identity/condition pairs are resumed rather than recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from projects.evoseg.temporal_dense_prompt.orchestrate import (
    PY_SA,
    PY_SAM,
    REPO,
    VAL,
    pixel_command,
)


def write_status(root: Path, state: str, active: list[dict], **extra) -> None:
    old = json.loads((root / "STATUS.json").read_text()) if (root / "STATUS.json").is_file() else {}
    value = {
        **old,
        "state": state,
        "phase": "recover_pilot64_pixel_oom",
        "updated_at": datetime.now().astimezone().isoformat(),
        "recovery_pid": os.getpid(),
        "active_processes": active,
        **extra,
    }
    temporary = root / "STATUS.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(root / "STATUS.json")


def run(args) -> int:
    root = Path(args.run_dir).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(args.shards) if args.shards else list(range(args.num_shards))
    recovery_shards = tuple(pending)
    running: dict[int, tuple[int, subprocess.Popen, object]] = {}
    returncodes: dict[int, int] = {}
    started = time.monotonic()

    while pending or running:
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            shard = pending.pop(0)
            command = pixel_command(
                VAL,
                root / "pilot64/dense",
                root / "pilot64/pixel",
                0,
                64,
                shard,
                args.num_shards,
            )
            handle = (logs / f"pilot_pixel_recovery_{shard}.log").open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(
                [PY_SAM, "-m", *command],
                cwd=REPO,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (shard, process, handle)

        active = [
            {"gpu": gpu, "shard": shard, "pid": process.pid}
            for gpu, (shard, process, _) in sorted(running.items())
        ]
        write_status(
            root,
            "running",
            active,
            recovery_shards_complete=len(returncodes),
            recovery_shards_planned=len(recovery_shards),
            recovery_elapsed_seconds=time.monotonic() - started,
            sam31_workers_per_gpu=1,
        )
        time.sleep(args.poll_seconds)
        for gpu, (shard, process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            returncodes[shard] = code
            del running[gpu]

    if any(returncodes.values()):
        write_status(root, "failed", [], recovery_returncodes=returncodes)
        raise RuntimeError(f"pixel recovery workers failed: {returncodes}")

    summary_command = [
        PY_SA,
        "-m",
        "projects.evoseg.temporal_dense_prompt.summarize",
        "--dense-root",
        str(root / "pilot64/dense"),
        "--pixel-root",
        str(root / "pilot64/pixel"),
        "--output",
        str(root / "pilot64/summary"),
        "--phase",
        "pilot",
    ]
    subprocess.run(summary_command, cwd=REPO, check=True)
    summary = json.loads((root / "pilot64/summary/summary.json").read_text())
    if not summary["pixel_coverage"]["complete"]:
        write_status(root, "failed", [], pilot=summary, recovery_returncodes=returncodes)
        raise RuntimeError(f"pixel recovery remains incomplete: {summary['pixel_coverage']}")

    subprocess.run(
        [
            PY_SA,
            "-m",
            "projects.evoseg.temporal_dense_prompt.finalize",
            "--run-dir",
            str(root),
            "--docs",
            str(Path(REPO) / "docs/temporal_dense_prompt"),
            "--manifest",
            VAL,
        ],
        cwd=REPO,
        check=True,
    )
    write_status(
        root,
        "complete",
        [],
        decision=summary["decision"],
        pilot=summary,
        recovery_returncodes=returncodes,
        recovery_elapsed_seconds=time.monotonic() - started,
        recovered_from="two_sam31_workers_per_gpu_cuda_oom",
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shards", nargs="+", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

"""Persistent 8-hour smoke-to-full experiment queue and report updater."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def update_status(path: Path, **updates):
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(updates)
    status["updated_at"] = now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_logged(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return process, handle


def summarize(worktree, run_dir):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "projects.evoseg.temporal_seg.summarize",
            "--run-dir",
            str(run_dir),
        ],
        cwd=worktree,
        check=False,
    )


def monitor(process, worktree, run_dir, deadline, interval=900):
    next_report = 0.0
    snapshot_written = False
    while process.poll() is None:
        current = time.time()
        if current >= next_report:
            summarize(worktree, run_dir)
            next_report = current + interval
        if current >= deadline and not snapshot_written:
            summarize(worktree, run_dir)
            update_status(
                run_dir / "STATUS.json",
                first_round_snapshot_at=now(),
                first_round_hours=8,
                child_continues_after_snapshot=True,
            )
            snapshot_written = True
        time.sleep(30)
    return process.returncode, snapshot_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    worktree = Path(args.worktree).resolve()
    data_root = Path(args.data_root).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    status_path = run_dir / "STATUS.json"
    started = time.time()
    deadline = started + args.hours * 3600
    update_status(
        status_path,
        state="waiting_for_long_rvos",
        supervisor_pid=os.getpid(),
        assigned_gpu=args.device,
        start_time=now(),
        first_round_deadline=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        blockers=[],
    )

    prepared = data_root / "valid" / ".prepared_at"
    while not prepared.exists():
        if time.time() >= deadline:
            summarize(worktree, run_dir)
            update_status(
                status_path,
                state="waiting_for_long_rvos_after_8h_snapshot",
                blockers=["Long-RVOS validation archive not extracted"],
            )
            deadline = float("inf")
        time.sleep(30)

    manifest_path = run_dir / "manifest.json"
    command = [
        sys.executable,
        "-m",
        "projects.evoseg.temporal_seg.build_manifest",
        "--metadata",
        str(data_root / "valid" / "meta_expressions.json"),
        "--image-root",
        str(data_root / "valid" / "JPEGImages"),
        "--annotation-root",
        str(data_root / "valid" / "Annotations"),
        "--output",
        str(manifest_path),
        "--require-files",
    ]
    manifest_log = (run_dir / "logs" / "manifest.log").open("a")
    result = subprocess.run(
        command,
        cwd=worktree,
        stdout=manifest_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    manifest_log.close()
    if result.returncode:
        update_status(
            status_path,
            state="blocked_manifest",
            blockers=["Long-RVOS extracted files did not pass manifest validation"],
        )
        summarize(worktree, run_dir)
        return result.returncode

    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    common = [
        sys.executable,
        "-m",
        "projects.evoseg.temporal_seg.runner",
        "--manifest",
        str(manifest_path),
        "--run-dir",
        str(run_dir),
        "--model",
        args.model,
        "--device",
        str(args.device),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    update_status(status_path, state="smoke_running")
    smoke, smoke_handle = run_logged(
        common + ["--max-videos", "2", "--phase", "smoke"],
        run_dir / "logs" / "smoke.log",
        env,
    )
    update_status(status_path, inference_pid=smoke.pid)
    smoke_rc, snapshot_written = monitor(smoke, worktree, run_dir, deadline)
    smoke_handle.close()
    summarize(worktree, run_dir)

    quality_log = (run_dir / "logs" / "quality_check.log").open("a")
    quality = subprocess.run(
        [
            sys.executable,
            "-m",
            "projects.evoseg.temporal_seg.quality_check",
            "--run-dir",
            str(run_dir),
        ],
        cwd=worktree,
        stdout=quality_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    quality_log.close()
    if quality.returncode:
        update_status(
            status_path,
            state="smoke_quality_failed",
            smoke_return_code=smoke_rc,
            blockers=["two-video smoke quality checks failed; full queue not started"],
        )
        summarize(worktree, run_dir)
        return quality.returncode

    update_status(status_path, state="full_running", smoke_return_code=smoke_rc)
    full, full_handle = run_logged(
        common + ["--phase", "full"], run_dir / "logs" / "full.log", env
    )
    update_status(status_path, inference_pid=full.pid)
    full_rc, full_snapshot = monitor(full, worktree, run_dir, deadline)
    full_handle.close()
    summarize(worktree, run_dir)
    update_status(
        status_path,
        state="complete" if full_rc == 0 else "complete_with_failures",
        full_return_code=full_rc,
        inference_pid=None,
        first_round_snapshot_written=snapshot_written or full_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

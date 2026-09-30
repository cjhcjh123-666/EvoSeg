"""Persist ProcessVIRST task status without coupling it to model processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def session_active(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def snapshot(root: Path, sessions: list[str], planned_steps: int) -> dict:
    tasks = []
    for name in sessions:
        seed = name.rsplit("seed", 1)[-1]
        output = root / f"pilot_train_seed{seed}"
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        completed = jsonl_rows(output / "training.jsonl")
        tasks.append(
            {
                "session": name,
                "seed": int(seed),
                "active": session_active(name),
                "completed_steps": completed,
                "planned_steps": planned_steps,
                "status": summary.get("status") if summary else ("running" if session_active(name) else "pending_or_failed"),
                "summary": summary,
            }
        )
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "run_root": str(root.resolve()),
        "tasks": tasks,
        "completed_steps": sum(task["completed_steps"] for task in tasks),
        "planned_steps": planned_steps * len(tasks),
        "all_finished": all(task["status"] == "success" for task in tasks),
    }


def progress_markdown(value: dict) -> str:
    lines = [
        "# ProcessVIRST progress",
        "",
        f"Updated: {value['updated_at']}",
        "",
        f"Training updates: {value['completed_steps']}/{value['planned_steps']}",
        "",
        "| Seed | Status | Updates | Session |",
        "|---:|---|---:|---|",
    ]
    for task in value["tasks"]:
        lines.append(
            f"| {task['seed']} | {task['status']} | "
            f"{task['completed_steps']}/{task['planned_steps']} | `{task['session']}` |"
        )
    lines.extend(["", "No final scientific conclusion is assigned before evaluation.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session", action="append", required=True)
    parser.add_argument("--planned-steps", type=int, required=True)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--progress-interval", type=int, default=7200)
    args = parser.parse_args()
    last_progress = 0.0
    while True:
        value = snapshot(args.run_root, args.session, args.planned_steps)
        atomic_write(args.run_root / "STATUS.json", json.dumps(value, indent=2) + "\n")
        now = time.monotonic()
        if now - last_progress >= args.progress_interval or value["all_finished"]:
            atomic_write(args.run_root / "PROGRESS.md", progress_markdown(value))
            last_progress = now
        if value["all_finished"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

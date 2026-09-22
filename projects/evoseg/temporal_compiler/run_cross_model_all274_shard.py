"""Gate and run one all-274 InstructSeg/VIRST shard on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MODELS = {
    "Sa2VA-Qwen3-VL-4B",
    "InstructSeg",
    "VIRST",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_pilot_overlap(value: dict) -> None:
    if value.get("expected_paired_objects") != 64:
        raise RuntimeError("pilot did not select exactly 64 paired objects")
    if set(value.get("successful_models", [])) != EXPECTED_MODELS:
        raise RuntimeError(
            "pilot did not complete all required model families: "
            f"{value.get('successful_models')}"
        )
    completed = value.get("complete_paired_objects_by_model", {})
    missing = value.get("missing_paired_objects_by_model", {})
    for model in EXPECTED_MODELS:
        if completed.get(model) != 64:
            raise RuntimeError(f"pilot has incomplete paired objects for {model}")
        if missing.get(model) != []:
            raise RuntimeError(f"pilot has missing paired objects for {model}")
    if value.get("shared_complete_paired_objects") != 64:
        raise RuntimeError("pilot shared complete-object intersection is not 64")
    if value.get("all_models_have_identical_complete_object_set") is not True:
        raise RuntimeError("pilot model object sets are not identical")


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def directory_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in path.glob("*/*") if child.is_dir())


def run(args) -> int:
    pilot_overlap_path = Path(args.pilot_overlap).resolve()
    overlap = json.loads(pilot_overlap_path.read_text())
    validate_pilot_overlap(overlap)

    repo = Path(args.repo).resolve()
    instruct_run_dir = Path(args.instruct_run_dir).resolve()
    virst_run_dir = Path(args.virst_run_dir).resolve()
    status_path = Path(args.status_json).resolve()
    instruct_runner = repo / "projects/evoseg/temporal_compiler/run_instructseg_long_rvos_eval.sh"
    virst_runner = repo / "projects/evoseg/temporal_compiler/run_virst_long_rvos_eval.sh"
    for required in (
        instruct_runner,
        virst_runner,
        instruct_run_dir / "expression_mapping.json",
        instruct_run_dir / "refyoutube_input.json",
        virst_run_dir / "dataset_root/expression_mapping.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    instruct_command = [str(instruct_runner), str(args.gpu), str(instruct_run_dir)]
    virst_command = [
        str(virst_runner),
        str(args.gpu),
        str(virst_run_dir / "dataset_root"),
        str(virst_run_dir),
        f"all274_s{args.shard_index}",
        str(args.master_port),
    ]
    environment = os.environ.copy()
    environment["EVOSEG_ROOT"] = str(repo)
    started_at = time.monotonic()
    instruct_process = subprocess.Popen(instruct_command, env=environment)
    virst_process = subprocess.Popen(virst_command, env=environment)

    def status(state: str) -> dict:
        return {
            "state": state,
            "updated_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started_at,
            "gpu": args.gpu,
            "shard_index": args.shard_index,
            "pilot_overlap": str(pilot_overlap_path),
            "pilot_gate_validated": True,
            "instructseg": {
                "pid": instruct_process.pid,
                "returncode": instruct_process.poll(),
                "completed_expression_directories": directory_count(
                    instruct_run_dir / "output/Annotations"
                ),
                "prediction_records": line_count(
                    instruct_run_dir / "predictions.jsonl"
                ),
                "command": instruct_command,
            },
            "virst": {
                "pid": virst_process.pid,
                "returncode": virst_process.poll(),
                "frame_audit_records": line_count(virst_run_dir / "frame_audit.jsonl"),
                "prediction_records": line_count(virst_run_dir / "predictions.jsonl"),
                "command": virst_command,
            },
        }

    atomic_json(status_path, status("running"))
    while instruct_process.poll() is None or virst_process.poll() is None:
        atomic_json(status_path, status("running"))
        time.sleep(args.poll_seconds)
    final_state = (
        "complete"
        if instruct_process.returncode == 0 and virst_process.returncode == 0
        else "complete_with_failures"
    )
    atomic_json(status_path, status(final_state))
    return 0 if final_state == "complete" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, choices=range(4), required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--pilot-overlap", required=True)
    parser.add_argument("--instruct-run-dir", required=True)
    parser.add_argument("--virst-run-dir", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--repo", default="/tmp/EvoSeg-temporal-compiler-sam31")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

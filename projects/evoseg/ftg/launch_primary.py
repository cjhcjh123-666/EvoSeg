"""Detach a single eight-GPU primary run without modifying existing GPU jobs."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--wall-limit-hours", type=float, default=0.0)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--development-count", type=int, default=32)
    parser.add_argument("--evaluate-after-resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.processes <= 8 or min(args.frame_budget, args.epochs, args.evaluate_every) < 1 or min(args.max_updates, args.wall_limit_hours, args.development_count) < 0:
        parser.error("invalid process/frame/update count")
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    launch_path = args.output / "LAUNCH.json"
    if launch_path.exists() or (args.output / "training.jsonl").exists():
        raise RuntimeError("existing launch/run detected; use a new run directory, do not overwrite it")
    if args.manifest:
        prepared = json.loads(args.manifest.read_text())
        if prepared["frame_budget"] != args.frame_budget or prepared.get("purpose") != "main_training":
            raise RuntimeError("incompatible prepared training manifest")
        destination = args.output / "manifest.json"
        if destination.exists():
            raise RuntimeError("refusing to overwrite an existing manifest")
        shutil.copyfile(args.manifest, destination)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    subprocess.run([sys.executable, "-m", "projects.evoseg.ftg.train_primary", "--prepare-only",
                    "--output", str(args.output), "--frame-budget", str(args.frame_budget)],
                   cwd=root, env=environment, check=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.processes}",
               "-m", "projects.evoseg.ftg.train_primary", "--output", str(args.output),
               "--frame-budget", str(args.frame_budget), "--max-updates", str(args.max_updates),
               "--epochs", str(args.epochs), "--wall-limit-hours", str(args.wall_limit_hours),
               "--evaluate-every", str(args.evaluate_every), "--development-count", str(args.development_count)]
    if args.resume:
        if not args.resume.is_file():
            raise RuntimeError("resume checkpoint does not exist")
        command.extend(["--resume", str(args.resume)])
    if args.evaluate_after_resume:
        command.append("--evaluate-after-resume")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    with (args.output / "console.log").open("a") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    payload = {"pid": process.pid, "command": command, "revision": revision, "working_directory": str(root),
               "launched_at_unix": time.time(), "processes": args.processes, "ablation_sweep": False}
    launch_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

"""Launch the six controlled FTG variants on separate visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .interface import FTG_VARIANTS
from .public_video_data import build_public_video_manifest


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty run directory: {args.output}")
    if len(args.gpus) < len(args.variants):
        raise ValueError("one GPU is required for each concurrently launched variant")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_public_video_manifest(
        args.long_root,
        args.mevis_root,
        long_per_type=args.long_per_type,
        mevis_count=args.mevis_count,
        frame_budget=args.frame_budget,
        seed=args.data_seed,
        validation_fraction=args.validation_fraction,
    )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    launches = []
    for variant, physical_gpu in zip(
        args.variants, args.gpus[: len(args.variants)], strict=True
    ):
        run_dir = args.output / variant
        run_dir.mkdir()
        command = [
            sys.executable,
            "-m", "projects.evoseg.ftg.train_pilot",
            "--variant", variant,
            "--output", str(run_dir),
            "--manifest", str(manifest_path),
            "--frame-budget", str(args.frame_budget),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--data-seed", str(args.data_seed),
            "--device", "0",
            "--selection-loss-weight", str(args.selection_loss_weight),
            "--query-policy", args.query_policy,
            "--sam-visual-chunk-size", str(args.sam_visual_chunk_size),
            "--sam-decode-chunk-size", str(args.sam_decode_chunk_size),
        ]
        if args.evaluate_initial:
            command.append("--evaluate-initial")
        if args.eval_orders:
            command.extend(["--eval-orders", *args.eval_orders])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        log_path = run_dir / "console.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        launches.append(
            {
                "variant": variant,
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
        "manifest": str(manifest_path),
        "frame_budget": args.frame_budget,
        "epochs": args.epochs,
        "selection_loss_weight": args.selection_loss_weight,
        "query_policy": args.query_policy,
        "sam_visual_chunk_size": args.sam_visual_chunk_size,
        "sam_decode_chunk_size": args.sam_decode_chunk_size,
        "launches": launches,
    }
    (args.output / "launch_state.json").write_text(json.dumps(state, indent=2) + "\n")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=FTG_VARIANTS, default=FTG_VARIANTS)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--long-per-type", type=int, default=8)
    parser.add_argument("--mevis-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--selection-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--query-policy", choices=("predicted_score", "fixed_slot"),
        default="predicted_score",
    )
    parser.add_argument("--sam-visual-chunk-size", type=int, default=4)
    parser.add_argument("--sam-decode-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--evaluate-initial", action="store_true")
    parser.add_argument(
        "--eval-orders", nargs="+", choices=("original", "shuffle", "reverse"),
        default=("original",),
    )
    parser.add_argument(
        "--long-root", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos/train"),
    )
    parser.add_argument(
        "--mevis-root", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/train"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

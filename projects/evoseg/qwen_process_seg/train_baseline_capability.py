"""Train the preregistered QwenSeg-SAM31 capability baseline.

This is deliberately a capability/overfit gate rather than a benchmark trainer.  It
uses the final SAM3.1 mask logits for BCE + Dice, keeps every SAM3.1 parameter
frozen, and records the complete gradient chain needed before process modelling is
allowed to begin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import get_peft_model_state_dict

from .baseline_model import QwenSegSAM31
from .long_rvos import LongRVOSCapabilityDataset, build_capability_manifest


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return squared.sqrt().item()


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest["records"]) != args.samples:
            raise RuntimeError("existing manifest sample count differs from requested count")
    else:
        manifest = build_capability_manifest(args.data_root, args.samples, args.data_seed)
        _atomic_json(manifest_path, manifest)
    dataset = LongRVOSCapabilityDataset(manifest)

    started = time.time()
    status = {
        "schema_version": 1,
        "stage": "baseline_capability",
        "status": "loading",
        "pid": os.getpid(),
        "sample_count": len(dataset),
        "epochs": args.epochs,
        "total_updates": len(dataset) * args.epochs,
        "completed_updates": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(args.output / "STATUS.json", status)

    model = QwenSegSAM31(
        args.qwen_checkpoint,
        args.sam_checkpoint,
        args.sam_repo,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    groups = model.parameter_groups()
    if not all(groups.values()):
        raise RuntimeError({key: len(value) for key, value in groups.items()})
    if model.executor.trainable_parameter_count() != 0:
        raise RuntimeError("SAM3.1 unexpectedly has trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["qwen_lora"], "lr": args.lora_lr},
            {"params": groups["fusion"], "lr": args.bridge_lr},
            {"params": groups["bridge"], "lr": args.bridge_lr},
        ],
        weight_decay=args.weight_decay,
    )
    parameter_counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    status.update({"status": "running", "trainable_parameters": parameter_counts})
    _atomic_json(args.output / "STATUS.json", status)

    logs: list[dict] = []
    epoch_losses: list[list[float]] = []
    latest_gradients: dict[str, float] = {}
    for epoch in range(args.epochs):
        losses = []
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        for sample_position, sample_index in enumerate(order):
            sample = dataset[sample_index]
            optimizer.zero_grad(set_to_none=True)
            before = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(sample["frames"], sample["expression"], sample["masks"])
            output["loss"].backward()
            latest_gradients = {name: _grad_norm(parameters) for name, parameters in groups.items()}
            sam_grad_parameters = sum(
                parameter.grad is not None for parameter in model.executor.assembled.parameters()
            )
            if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in latest_gradients.values()):
                raise RuntimeError(f"non-positive/non-finite trainable gradient: {latest_gradients}")
            if sam_grad_parameters != 0:
                raise RuntimeError("frozen SAM3.1 accumulated parameter gradients")
            optimizer.step()
            torch.cuda.synchronize(device)
            loss = output["loss"].detach().float().item()
            losses.append(loss)
            record = {
                "epoch": epoch,
                "sample_position": sample_position,
                "sample_index": sample_index,
                "video_id": sample["video_id"],
                "object_id": sample["object_id"],
                "expression_id": sample["expression_id"],
                "loss": loss,
                "loss_bce": output["loss_bce"].detach().float().item(),
                "loss_dice": output["loss_dice"].detach().float().item(),
                "mask_empty_fraction": output["mask_empty_fraction"].detach().float().item(),
                "mask_full_fraction": output["mask_full_fraction"].detach().float().item(),
                "prompt_cross_frame_std": output["prompt_cross_frame_std"].detach().float().item(),
                "visual_token_count": output["visual_token_count"],
                "gradient_norms": latest_gradients,
                "sam_parameter_grad_count": sam_grad_parameters,
                "elapsed_seconds": time.perf_counter() - before,
            }
            logs.append(record)
            with (args.output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            completed = epoch * len(dataset) + sample_position + 1
            status.update(
                {
                    "completed_updates": completed,
                    "last_loss": loss,
                    "last_gradients": latest_gradients,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "elapsed_seconds": time.time() - started,
                    "eta_seconds": (time.time() - started) / completed * (status["total_updates"] - completed),
                }
            )
            _atomic_json(args.output / "STATUS.json", status)
        epoch_losses.append(losses)

    first_epoch = _mean(epoch_losses[0])
    last_epoch = _mean(epoch_losses[-1])
    loss_gate_pass = (
        last_epoch < first_epoch
        if args.mode == "smoke"
        else (first_epoch - last_epoch) / first_epoch >= args.min_relative_improvement
    )
    result = {
        "status": "PASS" if loss_gate_pass else "FAIL",
        "gate": "QwenSeg-SAM31 baseline capability",
        "mode": args.mode,
        "required_relative_improvement": (
            0.0 if args.mode == "smoke" else args.min_relative_improvement
        ),
        "qwen_checkpoint": str(args.qwen_checkpoint),
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_checkpoint_sha256": hashlib.sha256(args.sam_checkpoint.read_bytes()).hexdigest(),
        "samples": len(dataset),
        "epochs": args.epochs,
        "updates": len(logs),
        "first_epoch_mean_loss": first_epoch,
        "last_epoch_mean_loss": last_epoch,
        "absolute_loss_change": last_epoch - first_epoch,
        "relative_loss_change": (last_epoch - first_epoch) / first_epoch,
        "last_gradient_norms": latest_gradients,
        "all_gradient_norms_positive": all(
            all(value > 0 for value in record["gradient_norms"].values()) for record in logs
        ),
        "sam_trainable_parameters": model.executor.trainable_parameter_count(),
        "sam_parameter_grad_count": max(record["sam_parameter_grad_count"] for record in logs),
        "mean_prompt_cross_frame_std": _mean([record["prompt_cross_frame_std"] for record in logs]),
        "mean_empty_fraction": _mean([record["mask_empty_fraction"] for record in logs]),
        "mean_full_fraction": _mean([record["mask_full_fraction"] for record in logs]),
        "trainable_parameters": parameter_counts,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "elapsed_seconds": time.time() - started,
    }
    if (
        not result["all_gradient_norms_positive"]
        or result["sam_trainable_parameters"] != 0
        or result["sam_parameter_grad_count"] != 0
        or result["mean_prompt_cross_frame_std"] <= 0
        or result["mean_empty_fraction"] >= 1
        or result["mean_full_fraction"] >= 1
    ):
        result["status"] = "FAIL"
    _atomic_json(args.output / "result.json", result)
    status.update({"status": result["status"], "completed_at": datetime.now(timezone.utc).isoformat()})
    _atomic_json(args.output / "STATUS.json", status)

    checkpoint = {
        "qwen_lora": {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(model.qwen).items()
        },
        "fusion": {key: value.detach().cpu() for key, value in model.fusion.state_dict().items()},
        "bridge": {key: value.detach().cpu() for key, value in model.bridge.state_dict().items()},
        "result": result,
    }
    torch.save(checkpoint, args.output / "lightweight_checkpoint.pt")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", choices=("smoke", "capability"), default="smoke")
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--bridge-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--qwen-checkpoint", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/models/SAM3.1-official-mirror/sam3.1_multiplex.pt"),
    )
    parser.add_argument(
        "--sam-repo", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/external/sam3"),
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos/train"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)

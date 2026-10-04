"""Train and evaluate one controlled FTG variant on the public video pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model_state_dict
from PIL import Image, ImageDraw, ImageOps

from .interface import FTG_VARIANTS
from .metrics import evaluate_masks, logits_to_masks
from .model import FTGQwenSAM31
from .public_video_data import PublicVideoPilotDataset, build_public_video_manifest


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    if not parameters:
        return 0.0
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return squared.sqrt().item()


def _switch_rate(indices: torch.Tensor) -> float:
    """Fraction of adjacent frames assigned to different SAM query slots."""
    indices = indices.detach().flatten()
    if indices.numel() < 2:
        return 0.0
    return (indices[1:] != indices[:-1]).float().mean().item()


def _ordered_sample(sample: dict, order: str, seed: int) -> tuple[list, torch.Tensor]:
    length = len(sample["frames"])
    if order == "original":
        indices = list(range(length))
    elif order == "reverse":
        indices = list(reversed(range(length)))
    elif order == "shuffle":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(length, generator=generator).tolist()
    else:
        raise ValueError(order)
    return [sample["frames"][index] for index in indices], sample["masks"][indices]


def _aggregate(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["overall"].append(record)
        groups[record["dataset"]].append(record)
        groups[f'{record["dataset"]}/{record["expression_type"]}'].append(record)
    result = {}
    for name, values in sorted(groups.items()):
        result[name] = {
            "count": len(values),
            **{
                metric: sum(value[metric] for value in values) / len(values)
                for metric in (
                    "j",
                    "f",
                    "jf",
                    "present_jf",
                    "false_accept",
                    "false_reject",
                    "oracle_j",
                    "oracle_f",
                    "oracle_jf",
                    "oracle_present_jf",
                    "query_selection_accuracy",
                    "matched_query_switch_rate",
                    "predicted_query_switch_rate",
                    "matched_mask_loss",
                    "predicted_mask_loss",
                )
            },
        }
    return result


def _save_qualitative(
    frames: list[Image.Image],
    target: torch.Tensor,
    prediction: np.ndarray,
    path: Path,
) -> None:
    panel_size = (224, 126)
    panels = []
    ground_truth = target.cpu().numpy().astype(bool)
    for frame_index, (frame, gt_mask, pred_mask) in enumerate(
        zip(frames, ground_truth, prediction, strict=True)
    ):
        panel = np.asarray(
            ImageOps.fit(frame.convert("RGB"), panel_size, Image.Resampling.BILINEAR)
        ).copy()
        gt = np.asarray(
            Image.fromarray(gt_mask).resize(panel_size, Image.Resampling.NEAREST)
        ).astype(bool)
        pred = np.asarray(
            Image.fromarray(pred_mask).resize(panel_size, Image.Resampling.NEAREST)
        ).astype(bool)
        colors = np.zeros_like(panel)
        colors[gt & ~pred] = (0, 255, 0)
        colors[pred & ~gt] = (255, 0, 0)
        colors[gt & pred] = (255, 255, 0)
        active = gt | pred
        panel[active] = (0.55 * panel[active] + 0.45 * colors[active]).astype(np.uint8)
        image = Image.fromarray(panel)
        ImageDraw.Draw(image).text((4, 4), f"t={frame_index}", fill=(255, 255, 255))
        panels.append(image)
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_size[0], rows * panel_size[1]))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


@torch.no_grad()
def evaluate(
    model: FTGQwenSAM31,
    dataset: PublicVideoPilotDataset,
    device: torch.device,
    order: str,
    seed: int,
    visual_dir: Path | None = None,
    visualize_count: int = 0,
) -> dict:
    model.eval()
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        frames, masks = _ordered_sample(sample, order, seed + index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(frames, sample["expression"], masks)
        prediction = logits_to_masks(output["predicted_mask_logits"], masks)
        metrics = evaluate_masks(masks, prediction)
        oracle_prediction = logits_to_masks(output["matched_mask_logits"], masks)
        oracle_metrics = {
            f"oracle_{name}": value
            for name, value in evaluate_masks(masks, oracle_prediction).items()
        }
        if visual_dir is not None and index < visualize_count:
            filename = (
                f'{index:03d}_{sample["dataset"]}_{sample["expression_type"]}_'
                f'{sample["video_id"]}_{sample["expression_id"]}.jpg'
            )
            _save_qualitative(frames, masks, prediction, visual_dir / filename)
        records.append(
            {
                "dataset": sample["dataset"],
                "video_id": sample["video_id"],
                "expression_id": sample["expression_id"],
                "expression_type": sample["expression_type"],
                "order": order,
                **metrics,
                **oracle_metrics,
                "query_selection_accuracy": (
                    output["predicted_query"] == output["matched_query"]
                ).float().mean().item(),
                "matched_query_switch_rate": _switch_rate(
                    output["matched_query"]
                ),
                "predicted_query_switch_rate": _switch_rate(
                    output["predicted_query"]
                ),
                "loss": output["loss"].float().item(),
                "matched_mask_loss": output["matched_mask_loss"].float().item(),
                "predicted_mask_loss": output["predicted_mask_loss"].float().item(),
                "gate_mean": output["gate_mean"].float().item(),
                "gate_std": output["gate_std"].float().item(),
                "prompt_cross_frame_std": output["prompt_cross_frame_std"].float().item(),
                "native_query_score_std": (
                    output["native_query_scores"].float().std().item()
                    if "native_query_scores" in output else None
                ),
                "identity_query_alignment_std": (
                    output["identity_query_alignment"].float().std().item()
                    if "identity_query_alignment" in output else None
                ),
            }
        )
    return {"order": order, "aggregate": _aggregate(records), "records": records}


def _load_or_build_manifest(args: argparse.Namespace) -> tuple[dict, Path]:
    manifest_path = args.manifest or args.output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("frame_budget") != args.frame_budget:
            raise RuntimeError("existing manifest frame budget differs from requested value")
    else:
        manifest = build_public_video_manifest(
            long_root=args.long_root,
            mevis_root=args.mevis_root,
            long_per_type=args.long_per_type,
            mevis_count=args.mevis_count,
            frame_budget=args.frame_budget,
            seed=args.data_seed,
            validation_fraction=args.validation_fraction,
        )
        _atomic_json(manifest_path, manifest)
    return manifest, manifest_path


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.sam_visual_chunk_size <= 0:
        raise ValueError("sam_visual_chunk_size must be positive")
    if args.sam_decode_chunk_size <= 0:
        raise ValueError("sam_decode_chunk_size must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = _load_or_build_manifest(args)
    train_data = PublicVideoPilotDataset(manifest, "train")
    validation_data = PublicVideoPilotDataset(manifest, "validation")
    if not train_data or not validation_data:
        raise RuntimeError("pilot requires non-empty train and validation partitions")

    started = time.time()
    status = {
        "schema_version": 1,
        "stage": "ftg_controlled_pilot",
        "variant": args.variant,
        "status": "loading",
        "pid": os.getpid(),
        "device": args.device,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "epochs": args.epochs,
        "total_updates": len(train_data) * args.epochs,
        "completed_updates": 0,
        "manifest_sha256": _sha256(manifest_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(args.output / "STATUS.json", status)

    model = FTGQwenSAM31(
        args.qwen_checkpoint,
        args.sam_checkpoint,
        args.sam_repo,
        variant=args.variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        qwen_pixels=args.qwen_pixels,
    ).to(device)
    model.selection_loss_weight = args.selection_loss_weight
    model.selection_loss_type = args.selection_loss_type
    model.query_policy = args.query_policy
    model.query_score_mode = args.query_score_mode
    if args.query_association_scale_init <= 0:
        raise ValueError("query association scale must be positive")
    model.query_association_logit_scale.data.fill_(
        math.log(args.query_association_scale_init)
    )
    model.match_scope = args.match_scope
    model.sam_interface = args.sam_interface
    model.native_residual_scale.data.fill_(args.native_residual_scale_init)
    if args.sam_interface == "native_factorized_residual":
        model.grounding.zero_active_output_projection()
    model.executor.visual_chunk_size = args.sam_visual_chunk_size
    model.executor.decode_chunk_size = args.sam_decode_chunk_size
    groups = model.parameter_groups()
    if not all(groups.values()):
        raise RuntimeError({key: len(value) for key, value in groups.items()})
    if model.executor.trainable_parameter_count() != 0:
        raise RuntimeError("SAM3.1 unexpectedly has trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["qwen_lora"], "lr": args.lora_lr},
            {"params": groups["grounding"], "lr": args.grounding_lr},
        ],
        weight_decay=args.weight_decay,
    )
    parameter_counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    status.update({"status": "running", "trainable_parameters": parameter_counts})
    _atomic_json(args.output / "STATUS.json", status)

    if args.evaluate_initial:
        initial = evaluate(model, validation_data, device, "original", args.seed)
        _atomic_json(args.output / "validation_initial.json", initial)

    model.train()
    model.executor.eval()
    training_log = args.output / "training.jsonl"
    if training_log.exists():
        training_log.unlink()
    all_losses = []
    latest_gradients = {}
    for epoch in range(args.epochs):
        order = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(args.seed + epoch)
        ).tolist()
        for position, sample_index in enumerate(order):
            sample = train_data[sample_index]
            optimizer.zero_grad(set_to_none=True)
            before = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(sample["frames"], sample["expression"], sample["masks"])
            output["loss"].backward()
            latest_gradients = {
                name: _grad_norm(parameters) for name, parameters in groups.items()
            }
            non_finite = any(
                not torch.isfinite(torch.tensor(value))
                for value in latest_gradients.values()
            )
            zero_gradient = any(value <= 0 for value in latest_gradients.values())
            expected_first_step_zero = (
                args.sam_interface == "native_factorized_residual"
                and epoch == 0
                and position == 0
                and latest_gradients["grounding"] > 0
            )
            if non_finite or (zero_gradient and not expected_first_step_zero):
                raise RuntimeError(f"non-positive/non-finite gradient: {latest_gradients}")
            sam_grad_count = sum(
                parameter.grad is not None for parameter in model.executor.assembled.parameters()
            )
            if sam_grad_count:
                raise RuntimeError("frozen SAM3.1 accumulated parameter gradients")
            optimizer.step()
            torch.cuda.synchronize(device)
            loss = output["loss"].detach().float().item()
            all_losses.append(loss)
            record = {
                "epoch": epoch,
                "position": position,
                "sample_index": sample_index,
                "dataset": sample["dataset"],
                "video_id": sample["video_id"],
                "expression_id": sample["expression_id"],
                "expression_type": sample["expression_type"],
                "loss": loss,
                "loss_bce": output["loss_bce"].detach().float().item(),
                "loss_dice": output["loss_dice"].detach().float().item(),
                "loss_selection": output["loss_selection"].detach().float().item(),
                "prompt_cross_frame_std": output["prompt_cross_frame_std"].detach().float().item(),
                "gate_mean": output["gate_mean"].detach().float().item(),
                "gradient_norms": latest_gradients,
                "native_residual_scale": (
                    output["native_residual_scale"].detach().float().item()
                    if "native_residual_scale" in output else None
                ),
                "native_query_score_std": (
                    output["native_query_scores"].detach().float().std().item()
                    if "native_query_scores" in output else None
                ),
                "identity_query_alignment_std": (
                    output["identity_query_alignment"].detach().float().std().item()
                    if "identity_query_alignment" in output else None
                ),
                "sam_parameter_grad_count": sam_grad_count,
                "elapsed_seconds": time.perf_counter() - before,
            }
            with training_log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            completed = epoch * len(train_data) + position + 1
            elapsed = time.time() - started
            status.update(
                {
                    "completed_updates": completed,
                    "last_loss": loss,
                    "last_gradients": latest_gradients,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / completed * (status["total_updates"] - completed),
                }
            )
            _atomic_json(args.output / "STATUS.json", status)

    validations = {}
    for order in args.eval_orders:
        visual_dir = args.output / "qualitative" if order == "original" else None
        validations[order] = evaluate(
            model,
            validation_data,
            device,
            order,
            args.seed,
            visual_dir=visual_dir,
            visualize_count=args.visualize_count,
        )
        _atomic_json(args.output / f"validation_{order}.json", validations[order])

    result = {
        "status": "COMPLETE",
        "variant": args.variant,
        "manifest_sha256": status["manifest_sha256"],
        "qwen_checkpoint": str(args.qwen_checkpoint),
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
        "frame_budget": manifest["frame_budget"],
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "epochs": args.epochs,
        "updates": status["total_updates"],
        "mean_training_loss": sum(all_losses) / len(all_losses),
        "last_gradient_norms": latest_gradients,
        "sam_trainable_parameters": model.executor.trainable_parameter_count(),
        "trainable_parameters": parameter_counts,
        "selection_loss_weight": args.selection_loss_weight,
        "selection_loss_type": args.selection_loss_type,
        "query_policy": args.query_policy,
        "query_score_mode": args.query_score_mode,
        "query_association_scale": (
            model.query_association_logit_scale.detach().float().exp().item()
        ),
        "match_scope": args.match_scope,
        "sam_interface": args.sam_interface,
        "native_residual_scale": model.native_residual_scale.detach().float().item(),
        "sam_visual_chunk_size": args.sam_visual_chunk_size,
        "sam_decode_chunk_size": args.sam_decode_chunk_size,
        "validation": {
            order: payload["aggregate"] for order, payload in validations.items()
        },
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(args.output / "result.json", result)
    checkpoint = {
        "variant": args.variant,
        "qwen_lora": {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(model.qwen).items()
        },
        "grounding": {
            key: value.detach().cpu() for key, value in model.grounding.state_dict().items()
        },
        "native_residual_scale": model.native_residual_scale.detach().cpu(),
        "query_association": {
            key: value.detach().cpu()
            for key, value in model.query_association.state_dict().items()
        },
        "query_association_logit_scale": (
            model.query_association_logit_scale.detach().cpu()
        ),
        "result": result,
    }
    torch.save(checkpoint, args.output / "lightweight_checkpoint.pt")
    status.update(
        {
            "status": "COMPLETE",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    _atomic_json(args.output / "STATUS.json", status)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=FTG_VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--long-per-type", type=int, default=8)
    parser.add_argument("--mevis-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--grounding-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--selection-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-loss-type",
        choices=("softmax_ce", "binary_objectness"),
        default="softmax_ce",
    )
    parser.add_argument(
        "--query-policy",
        choices=("predicted_score", "consistent_score", "fixed_slot"),
        default="predicted_score",
    )
    parser.add_argument(
        "--query-score-mode",
        choices=("native", "representation"),
        default="native",
    )
    parser.add_argument("--query-association-scale-init", type=float, default=1.0)
    parser.add_argument(
        "--match-scope", choices=("frame", "video"), default="frame",
    )
    parser.add_argument(
        "--sam-interface",
        choices=(
            "detector_grounding", "native_text_residual",
            "native_factorized_residual", "tracker_slot",
        ),
        default="detector_grounding",
    )
    parser.add_argument("--native-residual-scale-init", type=float, default=1e-3)
    parser.add_argument("--sam-visual-chunk-size", type=int, default=4)
    parser.add_argument("--sam-decode-chunk-size", type=int, default=4)
    parser.add_argument("--qwen-pixels", type=int, default=100352)
    parser.add_argument("--evaluate-initial", action="store_true")
    parser.add_argument("--visualize-count", type=int, default=6)
    parser.add_argument(
        "--eval-orders", nargs="+", choices=("original", "shuffle", "reverse"),
        default=("original",),
    )
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

"""Re-evaluate an FTG lightweight checkpoint with query-selection diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import set_peft_model_state_dict

from .interface import FTG_VARIANTS
from .model import FTGQwenSAM31
from .public_video_data import PublicVideoPilotDataset
from .train_pilot import _atomic_json, evaluate


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = args.variant or checkpoint["variant"]
    if variant != checkpoint["variant"]:
        raise ValueError("requested variant does not match checkpoint")
    manifest = json.loads(args.manifest.read_text())
    dataset = PublicVideoPilotDataset(manifest, "validation")
    model = FTGQwenSAM31(
        args.qwen_checkpoint,
        args.sam_checkpoint,
        args.sam_repo,
        variant=variant,
        qwen_pixels=args.qwen_pixels,
    ).to(device)
    model.selection_loss_weight = checkpoint.get("result", {}).get(
        "selection_loss_weight", 0.1
    )
    model.query_policy = checkpoint.get("result", {}).get(
        "query_policy", "predicted_score"
    )
    set_peft_model_state_dict(model.qwen, checkpoint["qwen_lora"])
    model.grounding.load_state_dict(checkpoint["grounding"])
    payload = {
        "variant": variant,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "orders": {},
    }
    for order in args.eval_orders:
        visual_dir = args.output / "qualitative" if order == "original" else None
        result = evaluate(
            model,
            dataset,
            device,
            order,
            args.seed,
            visual_dir=visual_dir,
            visualize_count=args.visualize_count,
        )
        payload["orders"][order] = result
        _atomic_json(args.output / f"validation_{order}.json", result)
    _atomic_json(args.output / "evaluation.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=FTG_VARIANTS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--qwen-pixels", type=int, default=100352)
    parser.add_argument("--visualize-count", type=int, default=12)
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
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    result = run(arguments)
    print(json.dumps({"variant": result["variant"], "orders": list(result["orders"])}, indent=2))

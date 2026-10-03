"""Audit gradient flow through the frozen official SAM3.1 mask decoder.

This script uses the official Object Multiplex checkpoint and its native
``extra_per_object_embeddings`` conditioning input.  The visual backbone is
executed under ``torch.no_grad``; the frozen mask decoder is then executed with
autograd enabled so mask loss can update only the injected grounding tensor.

It intentionally does not use the repository's vendored ``third_parts/sam3``
copy because that copy is the pre-Multiplex SAM3 tracker, not SAM3.1.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_OFFICIAL_REPO = Path(
    "/9950backfile/chenjiahui/evo_artifacts/external/sam3"
)
DEFAULT_CHECKPOINT = Path(
    "/9950backfile/chenjiahui/evo_artifacts/models/"
    "SAM3.1-official-mirror/sam3.1_multiplex.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(module: torch.nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def _tensor(value: Any) -> torch.Tensor:
    """Unwrap SAM3 NestedTensor values without importing a private type."""
    return value.tensors if hasattr(value, "tensors") else value


def extract_frozen_visual_features(
    assembled_model: torch.nn.Module,
    tracker: torch.nn.Module,
    image: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run only the official SAM3.1 visual path without constructing a graph."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # SAM3.1 shares the detector tri-head visual backbone with the tracker.
        outputs = assembled_model.detector.backbone.forward_image(
            image,
            need_sam3_out=False,
            need_interactive_out=False,
            need_propagation_out=True,
        )
        pyramid = outputs["sam2_backbone_out"]["backbone_fpn"]
        decoder = tracker.sam_mask_decoder
        high_res = [
            decoder.conv_s0(_tensor(pyramid[0])).detach(),
            decoder.conv_s1(_tensor(pyramid[1])).detach(),
        ]
        image_embedding = _tensor(pyramid[-1]).detach()
    return image_embedding, high_res


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official SAM3.1 gradient audit")
    if args.official_repo.resolve() not in map(Path, sys.path):
        sys.path.insert(0, str(args.official_repo.resolve()))

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256 and not args.allow_hash_mismatch:
        raise RuntimeError(
            "SAM3.1 checkpoint hash mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_hash}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    builder_output = io.StringIO()
    with contextlib.redirect_stdout(builder_output):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(args.checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    model = predictor.model
    freeze(model)
    tracker = model.tracker.model
    frozen_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    # A deterministic synthetic image is sufficient for the gate while still
    # exercising the *real* checkpointed visual backbone.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = torch.rand(
        1, 3, model.image_size, model.image_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    image_embedding, high_res = extract_frozen_visual_features(
        model, tracker, image
    )

    decoder = tracker.sam_mask_decoder
    prompt_dim = decoder.transformer_dim
    multiplex_count = decoder.multiplex_count
    grounding = torch.zeros(
        1,
        multiplex_count,
        prompt_dim,
        device=device,
        dtype=image_embedding.dtype,
        requires_grad=True,
    )

    # Keep decoder weights frozen but enable autograd with respect to grounding.
    decoder.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        decoded = decoder(
            image_embeddings=image_embedding,
            image_pe=tracker.get_propagation_dense_pe(),
            high_res_features=high_res,
            multimask_output=True,
            extra_per_object_embeddings=grounding,
        )
        mask_logits = decoded["masks"][:, 0, 0]
        height, width = mask_logits.shape[-2:]
        target = torch.zeros_like(mask_logits)
        target[..., height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
        loss = F.binary_cross_entropy_with_logits(mask_logits.float(), target.float())
    loss.backward()

    grad = grounding.grad
    if grad is None:
        raise RuntimeError("Grounding embedding has no gradient")
    grad_norm = grad.float().norm().item()
    grad_abs_max = grad.float().abs().max().item()
    finite = bool(torch.isfinite(grad).all().item())
    sam_parameter_grads = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    prompt_changes_pixels = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        perturbed = grounding.detach().clone()
        perturbed[:, 0] += 0.1
        changed = decoder(
            image_embeddings=image_embedding,
            image_pe=tracker.get_propagation_dense_pe(),
            high_res_features=high_res,
            multimask_output=True,
            extra_per_object_embeddings=perturbed,
        )["masks"][:, 0, 0]
        prompt_delta = (changed.float() - mask_logits.detach().float()).abs().mean().item()
        prompt_changes_pixels = prompt_delta > 0

    passed = (
        finite
        and grad_norm > 0
        and grad_abs_max > 0
        and trainable_parameters == 0
        and sam_parameter_grads == 0
        and prompt_changes_pixels
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "official_repo": str(args.official_repo),
        "official_commit": args.official_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "sam_image_size": model.image_size,
        "prompt_interface": "MultiplexMaskDecoder.extra_per_object_embeddings",
        "prompt_interface_status": "official_internal_not_public_wrapper",
        "prompt_shape": list(grounding.shape),
        "visual_embedding_shape": list(image_embedding.shape),
        "high_res_shapes": [list(tensor.shape) for tensor in high_res],
        "mask_logits_shape": list(mask_logits.shape),
        "loss": loss.item(),
        "grounding_grad_norm": grad_norm,
        "grounding_grad_abs_max": grad_abs_max,
        "grounding_grad_finite": finite,
        "prompt_perturbation_mask_logit_mae": prompt_delta,
        "sam_frozen_parameters": frozen_parameters,
        "sam_trainable_parameters": trainable_parameters,
        "sam_parameters_with_grad": sam_parameter_grads,
        "assembled_builder_reported_missing_keys": "Missing keys (" in builder_output.getvalue(),
        "assembled_builder_reported_unexpected_keys": "Unexpected keys (" in builder_output.getvalue(),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--official-commit", default="2345a4ad109ac29c569da749c91d84f10dc08c40")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    outcome = run(parse_args())
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)

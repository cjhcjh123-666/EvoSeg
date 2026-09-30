"""Minimal frozen-backbone ProcessVIRST SFT loop.

The loop deliberately reuses the official VIRST model, data preprocessing,
segmentation losses, SegPrompter, and SAM2 executor. Only the new process module
and its fusion LayerNorm are optimized in the first prototype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from projects.evoseg.process_virst.long_rvos_train_adapter import ORDER_PATTERN
from projects.evoseg.process_virst.virst_integration import install_process_virst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--groundmore-dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[11, 23, 42])
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--ordered-steps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--alignment-mode", choices=["monotonic", "global"], default="monotonic"
    )
    parser.add_argument("--disable-order-loss", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=32,
        help="Persist an atomic resumable checkpoint every N completed updates.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a periodic checkpoint made by this exact configuration.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--videochat-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """Capture all RNGs used by the single-worker official data pipeline."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    prompter,
    optimizer: AdamW,
    next_step: int,
    initial_parameters: dict[str, torch.Tensor],
) -> dict:
    return {
        "format_version": 1,
        "seed": args.seed,
        "next_step": next_step,
        "total_steps": args.warmup_steps + args.ordered_steps,
        "warmup_steps": args.warmup_steps,
        "ordered_steps": args.ordered_steps,
        "frames": args.frames,
        "learning_rate": args.learning_rate,
        "process_virst": prompter.conditioner.state_dict(),
        "fusion_norm": prompter.fusion_norm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "initial_parameters": initial_parameters,
        "rng_state": rng_state(),
        "process_config": {
            "alignment_mode": args.alignment_mode,
            "order_loss_enabled": not args.disable_order_loss,
        },
    }


def validate_resume(args: argparse.Namespace, state: dict) -> None:
    expected = {
        "seed": args.seed,
        "total_steps": args.warmup_steps + args.ordered_steps,
        "warmup_steps": args.warmup_steps,
        "ordered_steps": args.ordered_steps,
        "frames": args.frames,
        "learning_rate": args.learning_rate,
    }
    observed = {key: state.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"resume configuration mismatch: {observed=} {expected=}")
    expected_process = {
        "alignment_mode": args.alignment_mode,
        "order_loss_enabled": not args.disable_order_loss,
    }
    if state.get("process_config") != expected_process:
        raise ValueError(
            "resume ProcessVIRST configuration mismatch: "
            f"observed={state.get('process_config')} expected={expected_process}"
        )


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def permutation_for(text: str) -> str:
    value = hashlib.sha256(text.encode()).digest()[0]
    return "reverse" if value % 2 == 0 else "block_swap"


def build_model(args: argparse.Namespace):
    from model.builder import build_virst, load_checkpoint_virst
    from utils import conversation as conversation_lib
    from utils.argument import DataArguments, ModelArguments

    model_args = ModelArguments(
        tokenizer=str(args.tokenizer),
        videochat_checkpoint=str(args.videochat_checkpoint),
        sam2_checkpoint=str(args.sam2_checkpoint),
        version="qwen_2",
        vision_encode_type="video_image",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_patch_merge_type="spatial_nopad",
        mm_newline_position="nothing",
        mm_local_num_frames=4,
    )
    data_args = DataArguments(
        rvos_root=str(args.dataset_root),
        image_aspect_ratio="anyres_nopad",
        image_grid_pinpoints="(1x1),...,(6x6)",
        local_num_frames=4,
        seg_image_length=args.frames,
        seg_image_size=1024,
        frames_upbound=args.frames,
        frames_lowbound=args.frames,
    )
    config = AutoConfig.from_pretrained("model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), model_max_length=4096, padding_side="right"
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    tokenizer.add_tokens(["[SEG]"])
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = build_virst(config, model_args=model_args, checkpoint=str(args.checkpoint), seg_token_idx=seg_token_idx)
    model.resize_token_embeddings(len(tokenizer))
    config_values = {
        "max_num_pixels": data_args.max_num_pixels,
        "frame_grid_pinpoints": data_args.frame_grid_pinpoints,
        "image_grid_pinpoints": data_args.image_grid_pinpoints,
        "image_crop_resolution": data_args.image_crop_resolution,
        "image_split_resolution": data_args.image_split_resolution,
        "image_aspect_ratio": data_args.image_aspect_ratio,
        "frame_aspect_ratio": data_args.frame_aspect_ratio,
        "tokenizer_padding_side": tokenizer.padding_side,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "mm_newline_position": model_args.mm_newline_position,
        "mm_use_im_start_end": model_args.mm_use_im_start_end,
        "mm_use_im_patch_token": model_args.mm_use_im_patch_token,
    }
    for key, value in config_values.items():
        setattr(model.config, key, value)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = [
        name
        for name, _ in model.named_modules()
        if name.endswith(suffixes) and "layers" in name and "seg_model" not in name
    ]
    model = get_peft_model(
        model,
        LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model = load_checkpoint_virst(model, str(args.checkpoint))
    core_matches = [
        module
        for module in model.modules()
        if hasattr(module, "seg_token_idx")
        and hasattr(getattr(module, "model", None), "seg_prompter")
    ]
    if len(core_matches) != 1:
        raise RuntimeError(f"expected exactly one VIRST core, found {len(core_matches)}")
    core = core_matches[0]
    capture = install_process_virst(core, alignment_mode=args.alignment_mode)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in core.model.seg_prompter.conditioner.parameters():
        parameter.requires_grad_(True)
    for parameter in core.model.seg_prompter.fusion_norm.parameters():
        parameter.requires_grad_(True)
    return model, core, capture, tokenizer, data_args


def build_loader(
    tokenizer,
    data_args,
    steps: int,
    frames: int,
    long_root: Path,
    groundmore_root: Path | None,
):
    from data.base_dataset import collate_fn
    import data.rvos_dataset as rvos_module

    paths = {
        "mevis_long_train": (
            str(long_root / "mevis" / "train"),
            str(long_root / "mevis" / "train" / "meta_expressions.json"),
        )
    }
    dataset_names = ["mevis_long_train"]
    if groundmore_root is not None:
        paths["mevis_groundmore_train"] = (
            str(groundmore_root / "mevis" / "train"),
            str(groundmore_root / "mevis" / "train" / "meta_expressions.json"),
        )
        dataset_names.append("mevis_groundmore_train")
    original_paths = rvos_module._paths_for_root

    def process_paths(dataset, root):
        return paths[dataset] if dataset in paths else original_paths(dataset, root)

    rvos_module._DATA_INFO.update(paths)
    rvos_module._paths_for_root = process_paths
    RVOSDataset = rvos_module.RVOSDataset

    dataset = RVOSDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        num_classes_per_sample=1,
        samples_per_epoch=steps,
        num_frames_sample_range=f"{frames},{frames}",
        rvos_sample_ratio="||".join("1" for _ in dataset_names),
        rvos_seg_data="||".join(dataset_names),
        rvos_sample_policy="uniform",
        rvos_root=str(data_args.rvos_root),
        train=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model, core, capture, tokenizer, data_args = build_model(args)
    model.to(device=device, dtype=torch.bfloat16)
    core.model.seg_model.sam_prompt_encoder.to(torch.float32)
    model.eval()
    prompter = core.model.seg_prompter
    # Keep a true FP32 master copy for the small trainable modules. Converting
    # these parameters to BF16 makes 1e-5 AdamW updates smaller than the local
    # quantization interval and can silently leave beta/weights unchanged.
    prompter.conditioner.to(torch.float32)
    prompter.fusion_norm.to(torch.float32)
    prompter.conditioner.train()
    prompter.fusion_norm.train()
    named_parameters = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    parameters = [value for _, value in named_parameters]
    trainable_dtypes = sorted({str(value.dtype) for value in parameters})
    if trainable_dtypes != ["torch.float32"]:
        raise RuntimeError(f"trainable ProcessVIRST parameters are not FP32: {trainable_dtypes}")
    initial_parameters = {
        name: value.detach().float().cpu().clone() for name, value in named_parameters
    }
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    total_steps = args.warmup_steps + args.ordered_steps
    next_step = 0
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(args, resume_state)
        prompter.conditioner.load_state_dict(resume_state["process_virst"], strict=True)
        prompter.fusion_norm.load_state_dict(resume_state["fusion_norm"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        initial_parameters = resume_state["initial_parameters"]
        next_step = int(resume_state["next_step"])
        if not 0 <= next_step <= total_steps:
            raise ValueError(f"invalid resume next_step {next_step}/{total_steps}")
    loader = build_loader(
        tokenizer,
        data_args,
        total_steps - next_step,
        args.frames,
        args.dataset_root,
        args.groundmore_dataset_root,
    )
    # Dataset construction may touch RNG state. Restore only after it is fully
    # built so the first resumed sample is exactly the next random sample.
    if resume_state is not None:
        restore_rng_state(resume_state["rng_state"])
    records = []

    for step, batch in enumerate(loader, start=next_step):
        batch = move_to_device(batch, device)
        text = batch["questions"][0]
        is_ordered_stage = step >= args.warmup_steps
        verified_order = text.startswith("mevis_groundmore_train_") or bool(
            ORDER_PATTERN.search(text)
        )
        order_loss_enabled = (
            is_ordered_stage and verified_order and not args.disable_order_loss
        )
        capture.set_permutation(permutation_for(text) if order_loss_enabled else None)
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_masks"],
                images_clip=batch["images_clip"],
                images_sam=batch["images_sam"],
                image_ids=batch["image_ids"],
                labels=batch["labels"],
                modalities=batch["modalities"],
                gt_masks=batch["masks_list"],
                generation=False,
                seg_evaluate=False,
                use_cond_frames=False,
            )
            segmentation_loss = output["loss"]
            diagnostics = prompter.last_diagnostics
            if diagnostics is None:
                raise RuntimeError("missing ProcessVIRST diagnostics")
            if order_loss_enabled:
                if len(diagnostics.permuted) != 1:
                    raise RuntimeError("ordered sample did not compute its fixed permutation")
                permuted = next(iter(diagnostics.permuted.values()))
                order_loss = torch.relu(
                    0.2 - diagnostics.original.alignment_score + permuted.alignment_score
                ).mean()
            else:
                order_loss = segmentation_loss.new_zeros(())
            loss = segmentation_loss + order_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        record = {
            "step": step,
            "stage": "ordered_sft" if is_ordered_stage else "identity_warmup",
            "verified_order": verified_order,
            "permutation": diagnostics.permutations[0] if diagnostics.permutations else None,
            "loss": float(loss.detach()),
            "segmentation_loss": float(segmentation_loss.detach()),
            "mask_loss": float(output["mask_loss"].detach()),
            "order_loss": float(order_loss.detach()),
            "order_margin": (
                float(
                    (
                        diagnostics.original.alignment_score
                        - next(iter(diagnostics.permuted.values())).alignment_score
                    ).mean().detach()
                )
                if diagnostics.permuted
                else None
            ),
            "beta": float(diagnostics.original.beta.detach()),
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - step_start,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "trainable_dtype": trainable_dtypes[0],
            "alignment_mode": args.alignment_mode,
            "order_loss_enabled": not args.disable_order_loss,
        }
        records.append(record)
        with (args.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        completed = step + 1
        if args.save_every > 0 and (
            completed % args.save_every == 0 or completed == total_steps
        ):
            atomic_torch_save(
                checkpoint_payload(
                    args=args,
                    prompter=prompter,
                    optimizer=optimizer,
                    next_step=completed,
                    initial_parameters=initial_parameters,
                ),
                args.output_dir / "resume_latest.pt",
            )

    checkpoint = checkpoint_payload(
        args=args,
        prompter=prompter,
        optimizer=optimizer,
        next_step=total_steps,
        initial_parameters=initial_parameters,
    )
    atomic_torch_save(checkpoint, args.output_dir / "process_virst_sft.pt")
    squared_delta = 0.0
    max_abs_delta = 0.0
    changed_values = 0
    for name, value in named_parameters:
        delta = value.detach().float().cpu() - initial_parameters[name]
        squared_delta += float(delta.square().sum())
        max_abs_delta = max(max_abs_delta, float(delta.abs().max()))
        changed_values += int(delta.count_nonzero())
    summary = {
        "status": "success",
        "seed": args.seed,
        "steps": total_steps,
        "trainable_params": sum(value.numel() for value in parameters),
        "process_params": sum(value.numel() for value in prompter.conditioner.parameters()),
        "beta_final": float(prompter.conditioner.beta.detach()),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "trainable_dtypes": trainable_dtypes,
        "trainable_parameter_delta_l2": squared_delta**0.5,
        "trainable_parameter_delta_max_abs": max_abs_delta,
        "changed_trainable_values": changed_values,
        "alignment_mode": args.alignment_mode,
        "order_loss_enabled": not args.disable_order_loss,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

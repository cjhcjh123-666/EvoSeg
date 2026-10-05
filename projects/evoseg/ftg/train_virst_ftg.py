"""Train the small FTG interface on top of the public VIRST checkpoint.

The official VIRST checkout and its 7B/SAM2.1 weights stay frozen.  Only the
identity-conditioned gate in :mod:`projects.evoseg.ftg.virst_interface` is
optimized with public RVOS masks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from projects.evoseg.ftg.virst_interface import install_virst_ftg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--videochat-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mevis-root", type=Path)
    parser.add_argument("--long-adapter-root", type=Path)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-gate-delta", type=float, default=0.5)
    parser.add_argument(
        "--variant",
        choices=["ftg", "unconditioned_ftg", "scalar_ftg"],
        default="ftg",
    )
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.frames <= 0:
        parser.error("steps and frames must be positive")
    if args.mevis_root is None and args.long_adapter_root is None:
        parser.error("at least one public video training dataset is required")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def pad_video_frames_to_multiple(
    frames: torch.Tensor,
    multiple: int = 4,
) -> torch.Tensor:
    """Repeat the final VLM frame for VideoChat's fixed local-frame groups."""
    if frames.ndim < 1 or len(frames) == 0:
        raise ValueError("video frame tensor must be non-empty")
    if multiple <= 0:
        raise ValueError("frame multiple must be positive")
    missing = (-len(frames)) % multiple
    if missing == 0:
        return frames
    repeat_shape = (missing,) + (1,) * (frames.ndim - 1)
    padding = frames[-1:].repeat(repeat_shape)
    return torch.cat([frames, padding], dim=0)


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_model(args: argparse.Namespace):
    # These imports intentionally resolve against the unmodified public VIRST
    # checkout supplied on PYTHONPATH by the launcher.
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
        rvos_root="/unused/ftg-direct-paths",
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
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        model_args.version
    ]
    tokenizer.add_tokens(["[SEG]"])
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    model = build_virst(
        config,
        model_args=model_args,
        checkpoint=str(args.checkpoint),
        seg_token_idx=seg_token_idx,
    )
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

    suffixes = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
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
    wrapper = install_virst_ftg(
        model,
        variant=args.variant,
        max_gate_delta=args.max_gate_delta,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.composer.parameters():
        parameter.requires_grad_(True)
    return model, wrapper, tokenizer, data_args


def build_loader(args, tokenizer, data_args, remaining_steps: int) -> DataLoader:
    from data.base_dataset import collate_fn
    import data.rvos_dataset as rvos_module

    paths = {}
    dataset_names = []
    if args.mevis_root is not None:
        paths["mevis_public_train"] = (
            str(args.mevis_root),
            str(args.mevis_root / "meta_expressions_v2.json"),
        )
        dataset_names.append("mevis_public_train")
    if args.long_adapter_root is not None:
        long_root = args.long_adapter_root / "mevis" / "train"
        paths["mevis_long_train"] = (
            str(long_root),
            str(long_root / "meta_expressions.json"),
        )
        dataset_names.append("mevis_long_train")
    for image_root, metadata in paths.values():
        if not Path(image_root, "JPEGImages").is_dir():
            raise FileNotFoundError(f"missing JPEGImages: {image_root}")
        if not Path(metadata).is_file():
            raise FileNotFoundError(f"missing metadata: {metadata}")

    original_paths = rvos_module._paths_for_root

    def direct_paths(dataset, root):
        return paths[dataset] if dataset in paths else original_paths(dataset, root)

    rvos_module._DATA_INFO.update(paths)
    rvos_module._paths_for_root = direct_paths
    dataset = rvos_module.RVOSDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        num_classes_per_sample=1,
        samples_per_epoch=remaining_steps,
        num_frames_sample_range=f"{args.frames},{args.frames}",
        rvos_sample_ratio="||".join("1" for _ in dataset_names),
        rvos_seg_data="||".join(dataset_names),
        rvos_sample_policy="uniform",
        rvos_root=str(data_args.rvos_root),
        train=True,
    )
    def collate_with_short_video_padding(batch):
        padded = []
        for item in batch:
            item = dict(item)
            item["images_clip"] = pad_video_frames_to_multiple(
                item["images_clip"], multiple=4
            )
            padded.append(item)
        return collate_fn(padded, tokenizer=tokenizer)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_short_video_padding,
    )


def checkpoint_payload(args, wrapper, optimizer, next_step: int) -> dict:
    return {
        "format_version": 1,
        "method": "Factorized Temporal Grounding on public VIRST prompts",
        "args": {
            **vars(args),
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        "variant": args.variant,
        "max_gate_delta": args.max_gate_delta,
        "next_step": next_step,
        "composer": wrapper.composer.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, wrapper, tokenizer, data_args = build_model(args)
    model.to(device=device, dtype=torch.bfloat16)
    # SAM's prompt encoder is numerically required in FP32 by the public code.
    core = next(
        module for module in model.modules()
        if hasattr(module, "seg_token_idx")
        and hasattr(getattr(module, "model", None), "seg_model")
    )
    core.model.seg_model.sam_prompt_encoder.to(torch.float32)
    model.eval()
    wrapper.composer.to(torch.float32).train()
    parameters = [
        parameter for parameter in wrapper.composer.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    next_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["variant"] != args.variant or state["max_gate_delta"] != args.max_gate_delta:
            raise ValueError("resume method configuration differs")
        wrapper.composer.load_state_dict(state["composer"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        next_step = int(state["next_step"])
    if not 0 <= next_step <= args.steps:
        raise ValueError(f"invalid resume step {next_step}/{args.steps}")
    loader = build_loader(args, tokenizer, data_args, args.steps - next_step)

    for step, batch in enumerate(loader, start=next_step):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
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
            loss = output["loss"]
        diagnostics = wrapper.last_diagnostics
        if diagnostics is None:
            raise RuntimeError("FTG wrapper did not receive public VIRST prompts")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        identity_norm = diagnostics.identity.detach().float().norm(dim=-1).mean()
        correction_norm = (
            diagnostics.prompt.detach().float()
            - (diagnostics.identity + diagnostics.state).detach().float()
        ).norm(dim=-1).mean()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "mask_loss": float(output["mask_loss"].detach()),
            "grad_norm": float(grad_norm),
            "gate_mean": float(diagnostics.gate.detach().float().mean()),
            "gate_std": float(diagnostics.gate.detach().float().std()),
            "correction_to_identity": float(
                correction_norm / identity_norm.clamp_min(1e-6)
            ),
            "seconds": time.perf_counter() - step_started,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        with (args.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        completed = step + 1
        if args.save_every > 0 and (
            completed % args.save_every == 0 or completed == args.steps
        ):
            atomic_torch_save(
                checkpoint_payload(args, wrapper, optimizer, completed),
                args.output_dir / "resume_latest.pt",
            )

    final_state = checkpoint_payload(args, wrapper, optimizer, args.steps)
    atomic_torch_save(final_state, args.output_dir / "virst_ftg.pt")
    summary = {
        "status": "success",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "frames": args.frames,
        "trainable_params": sum(parameter.numel() for parameter in parameters),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

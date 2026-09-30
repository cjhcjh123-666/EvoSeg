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
    capture = install_process_virst(core)
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
    prompter.conditioner.train()
    prompter.fusion_norm.train()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    total_steps = args.warmup_steps + args.ordered_steps
    loader = build_loader(
        tokenizer,
        data_args,
        total_steps,
        args.frames,
        args.dataset_root,
        args.groundmore_dataset_root,
    )
    records = []

    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        text = batch["questions"][0]
        is_ordered_stage = step >= args.warmup_steps
        verified_order = text.startswith("mevis_groundmore_train_") or bool(
            ORDER_PATTERN.search(text)
        )
        capture.set_permutation(permutation_for(text) if is_ordered_stage and verified_order else None)
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
            if is_ordered_stage and verified_order:
                if diagnostics.permuted_score is None:
                    raise RuntimeError("ordered sample did not compute its fixed permutation")
                order_loss = torch.relu(
                    0.2 - diagnostics.original.alignment_score + diagnostics.permuted_score
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
            "permutation": diagnostics.permutation,
            "loss": float(loss.detach()),
            "segmentation_loss": float(segmentation_loss.detach()),
            "mask_loss": float(output["mask_loss"].detach()),
            "order_loss": float(order_loss.detach()),
            "order_margin": (
                float((diagnostics.original.alignment_score - diagnostics.permuted_score).mean().detach())
                if diagnostics.permuted_score is not None
                else None
            ),
            "beta": float(diagnostics.original.beta.detach()),
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - step_start,
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        records.append(record)
        with (args.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    checkpoint = {
        "seed": args.seed,
        "process_virst": prompter.conditioner.state_dict(),
        "fusion_norm": prompter.fusion_norm.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "process_virst_sft.pt")
    summary = {
        "status": "success",
        "seed": args.seed,
        "steps": total_steps,
        "trainable_params": sum(value.numel() for value in parameters),
        "process_params": sum(value.numel() for value in prompter.conditioner.parameters()),
        "beta_final": float(prompter.conditioner.beta.detach()),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

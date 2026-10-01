"""Fixed-64 CPG-VIRST architecture capability run.

The script intentionally stops at the training-only overfit gate.  It uses the
official VIRST builder/loss/SAM2 path and trains only the CPG modules installed
immediately before the official SegPrompter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoConfig, AutoTokenizer

from projects.evoseg.continuous_process_virst.losses import (
    object_discrimination_loss,
    order_contrast_loss,
    termination_entropy_regularizer,
)
from projects.evoseg.continuous_process_virst.groundmore_parser import (
    parse_sequential_query,
)
from projects.evoseg.continuous_process_virst.training_objects import (
    GroundMoReTrainingObjects,
)
from projects.evoseg.continuous_process_virst.virst_integration import (
    install_continuous_process_virst,
)
from projects.evoseg.process_virst.train_sft import move_to_device


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--groundmore-root", type=Path, required=True)
    parser.add_argument("--groundmore-source", type=Path, required=True)
    parser.add_argument("--groundmore-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11, choices=[11, 23, 42])
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--joint-steps", type=int, default=512)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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


def permutation_for(identity: str) -> str:
    return "reverse" if hashlib.sha256(identity.encode()).digest()[0] % 2 == 0 else "block_swap"


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
        rvos_root=str(args.long_root),
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
    model = build_virst(
        config,
        model_args=model_args,
        checkpoint=str(args.checkpoint),
        seg_token_idx=seg_token_idx,
    )
    model.resize_token_embeddings(len(tokenizer))
    for key, value in {
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
    }.items():
        setattr(model.config, key, value)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    data_args.image_processor = model.get_vision_tower().image_processor
    data_args.is_multimodal = True
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = [
        name for name, _ in model.named_modules()
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
    cores = [
        module for module in model.modules()
        if hasattr(module, "seg_token_idx") and hasattr(getattr(module, "model", None), "seg_prompter")
    ]
    if len(cores) != 1:
        raise RuntimeError(f"expected one VIRST core, found {len(cores)}")
    core = cores[0]
    capture = install_continuous_process_virst(core, max_states=6)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in core.model.seg_prompter.conditioner.parameters():
        parameter.requires_grad_(True)
    for parameter in core.model.seg_prompter.fusion_norm.parameters():
        parameter.requires_grad_(True)
    return model, core, capture, tokenizer, data_args


def build_exact_loader(tokenizer, data_args, frames: int, long_root: Path, ground_root: Path):
    """One deterministic epoch over each of the fixed 32+32 expressions.

    We use the official RVOS evaluation indexing only to make expression choice
    exact; images, masks, transforms, question formatting, and random-within-bin
    frame sampling remain the official VIRST dataset implementation.
    """

    from functools import partial

    from data.base_dataset import collate_fn
    import data.rvos_dataset as rvos_module

    paths = {
        "mevis_cpg_long": (
            str(long_root / "mevis" / "train"),
            str(long_root / "mevis" / "train" / "meta_expressions.json"),
        ),
        "mevis_cpg_groundmore": (
            str(ground_root / "mevis" / "train"),
            str(ground_root / "mevis" / "train" / "meta_expressions.json"),
        ),
    }
    original_paths = rvos_module._paths_for_root

    def process_paths(dataset, root):
        return paths[dataset] if dataset in paths else original_paths(dataset, root)

    rvos_module._DATA_INFO.update(paths)
    rvos_module._paths_for_root = process_paths

    def dataset(name: str):
        return rvos_module.RVOSDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            num_classes_per_sample=1,
            num_frames_sample_range=f"{frames},{frames}",
            rvos_sample_ratio="1",
            rvos_seg_data=name,
            rvos_sample_policy="uniform",
            rvos_root=str(data_args.rvos_root),
            train=False,
        )

    long_dataset = dataset("mevis_cpg_long")
    ground_dataset = dataset("mevis_cpg_groundmore")
    if len(long_dataset) != 32 or len(ground_dataset) != 32:
        raise RuntimeError(
            f"fixed overfit pool changed: long={len(long_dataset)}, ground={len(ground_dataset)}"
        )
    combined = ConcatDataset([long_dataset, ground_dataset])
    interleaved = [value for index in range(32) for value in (index, 32 + index)]
    return DataLoader(
        combined,
        batch_size=1,
        sampler=interleaved,
        num_workers=0,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )


def main() -> None:
    args = arguments()
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
    prompter.conditioner.to(torch.float32).train()
    prompter.fusion_norm.to(torch.float32).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {parameter.dtype for parameter in parameters} != {torch.float32}:
        raise RuntimeError("all trainable CPG parameters must retain FP32 masters")
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    total_steps = args.warmup_steps + args.joint_steps
    loader = build_exact_loader(
        tokenizer,
        data_args,
        args.frames,
        args.long_root,
        args.groundmore_root,
    )
    objects = GroundMoReTrainingObjects(args.groundmore_source, args.groundmore_metadata)
    records = []
    step = 0
    while step < total_steps:
      for batch in loader:
        if step >= total_steps:
            break
        batch = move_to_device(batch, device)
        question = batch["questions"][0]
        is_groundmore = question.startswith(
            ("mevis_groundmore_train_", "mevis_cpg_groundmore_")
        )
        surface_question = GroundMoReTrainingObjects._question(question)
        verified_order = (
            parse_sequential_query(surface_question).resolved if is_groundmore else True
        )
        capture.set_permutation(permutation_for(question) if verified_order else None)
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(
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
            diagnostics = prompter.last_diagnostics
            if diagnostics is None:
                raise RuntimeError("missing CPG diagnostics")
            if verified_order:
                if len(diagnostics.permuted) != 1:
                    raise RuntimeError("verified-order sample lacks fixed permutation")
                permuted = next(iter(diagnostics.permuted.values()))
                order_loss = order_contrast_loss(
                    diagnostics.original.alignment.score,
                    permuted.alignment.score,
                )
                order_margin = float(
                    (
                        diagnostics.original.alignment.score
                        - permuted.alignment.score
                    ).mean().detach()
                )
            else:
                order_loss = result["loss"].new_zeros(())
                order_margin = None
            object_batch = objects.load(
                batch["image_paths"][0], question, device
            ) if is_groundmore else None
            if object_batch is not None:
                object_masks, target = object_batch
                object_scores = prompter.score_training_object_masks(object_masks)
                object_loss = object_discrimination_loss(object_scores, target)
                object_correct = bool((object_scores.argmax(-1) == target).item())
                object_margin = float(
                    (
                        object_scores[0, target.item()]
                        - object_scores[0, torch.arange(object_scores.shape[1], device=device) != target.item()].max()
                    ).detach()
                )
            else:
                object_loss = result["loss"].new_zeros(())
                object_correct = None
                object_margin = None
            length_loss = termination_entropy_regularizer(
                diagnostics.original.alignment.terminal_prior
            )
            warmup = step < args.warmup_steps
            segmentation_loss = result["loss"]
            loss = object_loss + 0.5 * order_loss + 0.01 * length_loss
            if not warmup:
                loss = loss + segmentation_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        alignment = diagnostics.original.alignment
        record = {
            "step": step,
            "stage": "warmup" if warmup else "joint_sft",
            "question": question,
            "loss": float(loss.detach()),
            "segmentation_loss": float(segmentation_loss.detach()),
            "mask_loss": float(result["mask_loss"].detach()),
            "object_loss": float(object_loss.detach()),
            "object_correct": object_correct,
            "target_distractor_margin": object_margin,
            "verified_order": verified_order,
            "interval_loss": None,
            "interval_supervision": "disabled: official interval is process-span, not clause-span",
            "order_loss": float(order_loss.detach()),
            "order_margin": order_margin,
            "expected_process_length": float(alignment.expected_length.mean().detach()),
            "mean_state_duration": float(alignment.expected_duration.mean().detach()),
            "posterior_entropy": float(alignment.posterior_entropy.mean().detach()),
            "beta": float(diagnostics.original.beta.detach()),
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - step_start,
        }
        records.append(record)
        with (args.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        step += 1
    checkpoint = {
        "format_version": 1,
        "seed": args.seed,
        "steps": total_steps,
        "continuous_process": prompter.conditioner.state_dict(),
        "fusion_norm": prompter.fusion_norm.state_dict(),
        "interval_supervision": "disabled_official_process_span_not_clause_span",
    }
    temporary = args.output_dir / f".cpg_sft.{os.getpid()}.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(args.output_dir / "cpg_overfit.pt")
    summary = {
        "status": "success",
        "seed": args.seed,
        "steps": total_steps,
        "trainable_params": sum(parameter.numel() for parameter in parameters),
        "beta_final": float(prompter.conditioner.beta.detach()),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - start,
        "interval_supervision_enabled": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

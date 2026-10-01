"""Fixed-64 CAPG-v2 capability training with no validation-driven tuning."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from projects.evoseg.continuous_process_virst.groundmore_parser import parse_sequential_query
from projects.evoseg.continuous_process_virst.serialization import json_safe
from projects.evoseg.continuous_process_virst.train_overfit import (
    build_exact_loader,
    build_model as build_v1_model,
    permutation_for,
    set_seed,
)
from projects.evoseg.process_virst.train_sft import move_to_device

from .losses import (
    language_order_loss,
    language_overlap_loss,
    object_discrimination_loss,
    order_contrast_loss,
    process_envelope_loss,
    segment_alignment_loss,
    temporal_iou,
)
from .training_context import GroundMoReTrainingContext
from .virst_integration import install_content_adaptive_virst


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
    parser.add_argument("--transition-mode", choices=["adaptive", "global"], default="adaptive")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--videochat-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    model, core, old_capture, tokenizer, data_args = build_v1_model(args)
    old_capture.close()
    official = core.model.seg_prompter.official
    core.model.seg_prompter = official
    capture = install_content_adaptive_virst(
        core, max_states=6, transition_mode=args.transition_mode
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in core.model.seg_prompter.conditioner.parameters():
        parameter.requires_grad_(True)
    for parameter in core.model.seg_prompter.fusion_norm.parameters():
        parameter.requires_grad_(True)
    return model, core, capture, tokenizer, data_args


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "training.jsonl").exists():
        raise FileExistsError("output directory already contains training.jsonl")
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model, core, capture, tokenizer, data_args = build_model(args)
    model.to(device=device, dtype=torch.bfloat16)
    core.model.seg_model.sam_prompt_encoder.to(torch.float32)
    model.eval()
    prompter = core.model.seg_prompter
    prompter.conditioner.to(torch.float32).train()
    prompter.fusion_norm.to(torch.float32).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {parameter.dtype for parameter in parameters} != {torch.float32}:
        raise RuntimeError("all trainable CAPG parameters must use FP32 masters")
    trainable = sum(parameter.numel() for parameter in parameters)
    if trainable >= 5_000_000:
        raise RuntimeError(f"CAPG exceeds preregistered 5M parameter target: {trainable}")
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    loader = build_exact_loader(
        tokenizer, data_args, args.frames, args.long_root, args.groundmore_root
    )
    context_loader = GroundMoReTrainingContext(
        args.groundmore_source, args.groundmore_metadata
    )
    total_steps = args.warmup_steps + args.joint_steps
    step = 0
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            batch = move_to_device(batch, device)
            question = batch["questions"][0]
            is_groundmore = question.startswith(("mevis_groundmore_train_", "mevis_cpg_groundmore_"))
            surface = GroundMoReTrainingContext.question(question)
            verified_order = parse_sequential_query(surface).resolved if is_groundmore else True
            permutation = permutation_for(question) if verified_order else None
            capture.set_permutation(permutation)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
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
                    raise RuntimeError("missing CAPG diagnostics")
                output = diagnostics.original
                zero = result["loss"].new_zeros(())
                if verified_order:
                    permuted = next(iter(diagnostics.permuted.values()))
                    order_loss = order_contrast_loss(
                        output.alignment.score, permuted.alignment.score
                    )
                    order_margin = float(
                        (output.alignment.score - permuted.alignment.score).mean().detach()
                    )
                else:
                    order_loss, order_margin = zero, None
                training_context = context_loader.load(
                    batch["image_paths"][0], question, device
                ) if is_groundmore else None
                if training_context is not None:
                    envelope_loss = process_envelope_loss(
                        output.alignment.process_occupancy,
                        training_context.process_envelope,
                    )
                    envelope_tiou = float(
                        temporal_iou(
                            output.alignment.process_occupancy,
                            training_context.process_envelope,
                        ).mean().detach()
                    )
                else:
                    envelope_loss, envelope_tiou = zero, None
                if training_context is not None and training_context.object_masks is not None:
                    object_scores = prompter.score_training_object_masks(
                        training_context.object_masks
                    )
                    object_loss = object_discrimination_loss(
                        object_scores, training_context.target_index
                    )
                    object_correct = bool(
                        (object_scores.argmax(-1) == training_context.target_index).item()
                    )
                    distractor = torch.arange(object_scores.shape[1], device=device) != int(
                        training_context.target_index.item()
                    )
                    object_margin = float(
                        (
                            object_scores[0, training_context.target_index.item()]
                            - object_scores[0, distractor].max()
                        ).detach()
                    )
                    object_count = object_scores.shape[1]
                else:
                    object_loss, object_correct, object_margin, object_count = zero, None, None, None
                segment_loss, semantic_margin = segment_alignment_loss(
                    output.process_states,
                    output.frame_features,
                    output.alignment.posterior_process,
                    output.length_prior,
                    detach_posterior=False,
                )
                detached_segment_loss, _ = segment_alignment_loss(
                    output.process_states,
                    output.frame_features,
                    output.alignment.posterior_process,
                    output.length_prior,
                    detach_posterior=True,
                )
                lang_order = language_order_loss(
                    output.language_positions, output.length_prior
                )
                lang_overlap = language_overlap_loss(
                    output.language_attention, output.length_prior
                )
                auxiliary = (
                    object_loss
                    + 0.5 * envelope_loss
                    + 0.5 * segment_loss
                    + 0.5 * order_loss
                    + 0.1 * lang_order
                    + 0.05 * lang_overlap
                )
                warmup = step < args.warmup_steps
                total = auxiliary if warmup else auxiliary + result["loss"]
            total.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            alignment = output.alignment
            record = {
                "step": step,
                "stage": "warmup" if warmup else "joint_sft",
                "transition_mode": args.transition_mode,
                "question": question,
                "expression_id": batch["exp_id"][0] if batch["exp_id"] else None,
                "sampled_frame_indices": json_safe(batch["frame_ids"]),
                "loss": float(total.detach()),
                "segmentation_loss": float(result["loss"].detach()),
                "object_loss": float(object_loss.detach()),
                "object_correct": object_correct,
                "object_count": object_count,
                "target_distractor_margin": object_margin,
                "envelope_loss": float(envelope_loss.detach()) if training_context is not None else None,
                "envelope_tiou": envelope_tiou,
                "segment_alignment_loss": float(segment_loss.detach()),
                "segment_alignment_detached_loss": float(detached_segment_loss.detach()),
                "segment_semantic_margin": float(semantic_margin.detach()),
                "language_order_loss": float(lang_order.detach()),
                "language_overlap_loss": float(lang_overlap.detach()),
                "verified_order": verified_order,
                "permutation": permutation,
                "order_loss": float(order_loss.detach()),
                "order_margin": order_margin,
                "length_prior": output.length_prior.detach().float().cpu().tolist()[0],
                "halt_probabilities": output.halt_probabilities.detach().float().cpu().tolist()[0],
                "expected_process_length": float(alignment.expected_length.mean().detach()),
                "posterior_entropy": float(alignment.posterior_entropy.mean().detach()),
                "posterior_pre": alignment.posterior_pre.detach().float().cpu().tolist()[0],
                "posterior_process": alignment.posterior_process.detach().float().cpu().tolist()[0],
                "posterior_post": alignment.posterior_post.detach().float().cpu().tolist()[0],
                "process_occupancy": alignment.process_occupancy.detach().float().cpu().tolist()[0],
                "start_posterior": alignment.start_posterior.detach().float().cpu().tolist()[0],
                "end_posterior": alignment.end_posterior.detach().float().cpu().tolist()[0],
                "compatibility": output.compatibility.detach().float().cpu().tolist()[0],
                "language_attention": output.language_attention.detach().float().cpu().tolist()[0],
                "language_positions": output.language_positions.detach().float().cpu().tolist()[0],
                "beta": float(output.beta.detach()),
                "grad_norm": float(gradient),
                "seconds": time.perf_counter() - step_started,
            }
            with (args.output_dir / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                json.dumps(
                    {
                        "step": step,
                        "stage": record["stage"],
                        "mode": args.transition_mode,
                        "loss": record["loss"],
                        "segmentation_loss": record["segmentation_loss"],
                        "order_margin": order_margin,
                        "envelope_tiou": envelope_tiou,
                        "object_correct": object_correct,
                        "expected_N": record["expected_process_length"],
                        "beta": record["beta"],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
            step += 1
    checkpoint = {
        "format_version": 2,
        "seed": args.seed,
        "transition_mode": args.transition_mode,
        "steps": total_steps,
        "conditioner": prompter.conditioner.state_dict(),
        "fusion_norm": prompter.fusion_norm.state_dict(),
    }
    temporary = args.output_dir / f".capg.{os.getpid()}.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(args.output_dir / "capg_capability.pt")
    summary = {
        "status": "success",
        "seed": args.seed,
        "transition_mode": args.transition_mode,
        "steps": total_steps,
        "trainable_params": trainable,
        "beta_final": float(prompter.conditioner.beta.detach()),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

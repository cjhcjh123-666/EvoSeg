"""Main-model-only distributed training; no ablation sweep or test-set tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from .primary_model import PrimaryFTGSAM31
from .public_video_data import PublicVideoPilotDataset
from .metrics import evaluate_masks, logits_to_masks
from projects.evoseg.qwen_process_seg.frame_protocol import uniform_frame_indices


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def build_manifest(long_root, mevis_root, frame_budget, seed=42):
    records, rejected = [], []
    for name, root, metadata in (("long_rvos", long_root, "meta_expressions.json"), ("mevis_v2", mevis_root, "meta_expressions_v2.json")):
        videos = json.loads((root / metadata).read_text())["videos"]
        video_names = sorted(videos)
        random.Random(seed).shuffle(video_names)
        held_out = set(video_names[:max(1, math.ceil(len(video_names) * 0.05))])
        for video_id in sorted(videos):
            video = videos[video_id]
            indices = uniform_frame_indices(len(video["frames"]), frame_budget)
            frame_names = [video["frames"][i] for i in indices]
            paths = [root / "JPEGImages" / video_id / f"{frame}.jpg" for frame in frame_names]
            if not all(path.is_file() for path in paths):
                rejected.append({"dataset": name, "video_id": video_id, "reason": "missing_sampled_image"})
                continue
            for expression_id, expression in sorted(video["expressions"].items()):
                record = {"dataset": name, "split": "train", "video_id": video_id,
                          "expression_id": str(expression_id), "expression": expression["exp"],
                          "expression_type": expression.get("type", "motion").lower(),
                          "source_frame_indices": indices, "source_frame_names": frame_names,
                          "image_paths": list(map(str, paths)),
                          "pilot_partition": "validation" if video_id in held_out else "train"}
                if name == "long_rvos":
                    record["object_id"] = str(expression["obj_id"])
                    masks = [root / "Annotations" / video_id / record["object_id"] / f"{frame}.png" for frame in frame_names]
                    record["mask_paths"] = [str(path) if path.is_file() else None for path in masks]
                else:
                    record["annotation_ids"] = [str(i) for i in expression.get("anno_id", [])]
                    record["object_id"] = "+".join(map(str, expression.get("obj_id", [])))
                    record["mask_dictionary"] = str(root / "mask_dict.json")
                records.append(record)
    return {"schema_version": 1, "purpose": "main_training", "frame_budget": frame_budget,
            "seed": seed, "held_out_fraction": 0.05, "holdout_unit": "video",
            "records": records, "rejected_videos": rejected,
            "training_sources": [str(long_root), str(mevis_root)],
            "no_synthetic_annotations": True}


@torch.no_grad()
def evaluate_development(model, dataset, device, output, update, count=8):
    """Sampled train-holdout monitoring; never report as official benchmark J&F."""
    model.eval()
    # Interleave datasets so a short check does not contain only Long-RVOS.
    by_source = {}
    for index, record in enumerate(dataset.records):
        by_source.setdefault(record["dataset"], []).append(index)
    indices = []
    for position in range(count):
        for source in sorted(by_source):
            if position < len(by_source[source]) and len(indices) < count:
                indices.append(by_source[source][position])
    records = []
    for index in indices:
        sample = dataset[index]
        source_indices = torch.tensor(sample["source_frame_indices"], device=device).float()
        times = (source_indices - source_indices.min()) / (source_indices.max() - source_indices.min()).clamp_min(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, presence, diagnostics = model.predict(sample["frames"], sample["expression"], times)
        logits = torch.where(presence[:, None, None] > 0, logits, -32.0)
        prediction = logits_to_masks(logits, sample["masks"])
        record = {"dataset": sample["dataset"], "video_id": sample["video_id"], "expression_id": sample["expression_id"],
                  "expression": sample["expression"], **evaluate_masks(sample["masks"], prediction)}
        records.append(record)
        if len(records) <= 4:
            from .train_pilot import _save_qualitative
            _save_qualitative(sample["frames"], sample["masks"], prediction,
                              output / "qualitative" / f"update_{update:06d}_{len(records):02d}.jpg")
    result = {"purpose": "sampled_train_video_holdout_monitoring_not_official_benchmark", "update": update,
              "count": len(records), "mean_jf": sum(r["jf"] for r in records) / max(1, len(records)), "records": records}
    atomic_json(output / f"development_{update:06d}.json", result)
    model.train()
    return result


def run(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if rank == 0:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing["frame_budget"] != args.frame_budget:
                raise RuntimeError("existing manifest has a different frame budget")
        else:
            atomic_json(manifest_path, build_manifest(args.long_root, args.mevis_root, args.frame_budget))
        atomic_json(args.output / "STATUS.json", {"status": "LOADING", "world_size": world, "pid": os.getpid(), "configuration": vars(args) | {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}})
    if world > 1:
        dist.barrier()
    manifest = json.loads(manifest_path.read_text())
    data = PublicVideoPilotDataset(manifest, "train")
    development = PublicVideoPilotDataset(manifest, "validation")
    if not len(data):
        raise RuntimeError("no public training records")
    torch.manual_seed(args.seed)
    model = PrimaryFTGSAM31(args.foundation, args.sam_checkpoint, args.sam_repo, args.qwen_pixels).to(device)
    if model.executor.trainable_parameter_count():
        raise RuntimeError("pixel foundation must stay frozen")
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([{"params": groups["qwen_lora"], "lr": args.lora_lr}, {"params": groups["grounding"], "lr": args.grounding_lr}], weight_decay=0.01)
    start_update = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise RuntimeError("resume manifest mismatch")
        state = model.state_dict()
        for key, value in payload["trainable_state"].items():
            state[key].copy_(value)
        optimizer.load_state_dict(payload["optimizer"])
        start_update = payload["update"]
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    updates_per_epoch = math.ceil(len(sampler) / args.accumulation)
    total_updates = args.max_updates or updates_per_epoch * args.epochs
    status = {"status": "RUNNING", "stage": "primary_model_training", "world_size": world,
              "train_expressions": len(data), "held_out_expressions": sum(r["pilot_partition"] == "validation" for r in manifest["records"]),
              "frame_budget": args.frame_budget, "global_batch": world * args.accumulation,
              "total_updates": total_updates, "completed_updates": start_update,
              "initialization": model.initialization_report,
              "sam_checkpoint": str(args.sam_checkpoint), "sam_repo": str(args.sam_repo),
              "sam_interface": "official_sam31_interactive_sparse_prompt",
              "sam_trainable_parameters": 0,
              "trainable_parameters": {k: sum(p.numel() for p in v) for k, v in groups.items()},
              "started_at_unix": time.time(), "pid": os.getpid(),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    if rank == 0:
        atomic_json(args.output / "STATUS.json", status)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    wrapped.train()
    update = start_update
    stop = False
    for epoch in range(math.ceil(total_updates / updates_per_epoch) + 1):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for offset in range(0, len(indices), args.accumulation):
            scheduled_update = epoch * updates_per_epoch + offset // args.accumulation
            if scheduled_update < start_update:
                continue
            warmup = min(50, max(1, total_updates // 20))
            warmup_scale = min(1.0, (update + 1) / warmup)
            phase = max(0.0, (update - warmup) / max(1, total_updates - warmup))
            scale = warmup_scale * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, phase))))
            optimizer.param_groups[0]["lr"] = args.lora_lr * scale
            optimizer.param_groups[1]["lr"] = args.grounding_lr * scale
            batch_indices = indices[offset:offset + args.accumulation]
            record_losses = []
            for micro, sample_index in enumerate(batch_indices):
                sample = data[sample_index]
                source_indices = torch.tensor(sample["source_frame_indices"], device=device).float()
                times = (source_indices - source_indices.min()) / (source_indices.max() - source_indices.min()).clamp_min(1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = wrapped(sample["frames"], sample["expression"], sample["masks"], times)
                    loss = output["loss"] / len(batch_indices)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at {sample['dataset']}/{sample['video_id']}/{sample['expression_id']}")
                loss.backward()
                record_losses.append(output["loss"].detach().float())
            norms = {}
            for name, parameters in groups.items():
                squared = sum((p.grad.detach().float().square().sum() for p in parameters if p.grad is not None), torch.zeros((), device=device))
                norms[name] = squared.sqrt().item()
                if not math.isfinite(norms[name]) or norms[name] <= 0:
                    raise RuntimeError(f"invalid {name} gradients: {norms}")
            if any(p.grad is not None for p in model.executor.parameters()):
                raise RuntimeError("SAM3.1 unexpectedly accumulated gradients")
            torch.nn.utils.clip_grad_norm_([p for values in groups.values() for p in values], 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            mean_loss = torch.stack(record_losses).mean()
            if world > 1:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                elapsed = time.time() - started
                progress = {"update": update, "epoch": epoch, "loss": mean_loss.item(), "gradient_norms": norms,
                            "rank0_bce": output["bce"].detach().float().item(),
                            "rank0_dice": output["dice"].detach().float().item(),
                            "rank0_presence_fraction": output["predicted_presence_fraction"].detach().float().item(),
                            "rank0_foreground_fraction": output["predicted_foreground_fraction"].detach().float().item(),
                            "state_norm": output["state_norm"].detach().float().item(), "gate_mean": output["gate_mean"].detach().float().item(),
                            "learning_rates": [group["lr"] for group in optimizer.param_groups],
                            "elapsed_seconds": elapsed, "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                            "eta_seconds": elapsed / (update - start_update) * (total_updates - update)}
                with (args.output / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(progress) + "\n")
                status.update(progress | {"completed_updates": update})
                atomic_json(args.output / "STATUS.json", status)
                print(json.dumps(progress), flush=True)
                if update % args.save_every == 0 or update == total_updates:
                    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
                    payload = {"trainable_state": {key: value.detach().cpu() for key, value in model.state_dict().items() if key in trainable_names},
                               "optimizer": optimizer.state_dict(), "update": update, "status": status,
                               "manifest_sha256": status["manifest_sha256"], "rng_state": torch.get_rng_state(),
                               "cuda_rng_state": torch.cuda.get_rng_state()}
                    path = args.output / f"checkpoint_{update:06d}.pt"
                    temporary = path.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    os.replace(temporary, path)
                    if args.development_count:
                        development_result = evaluate_development(model, development, device, args.output, update, args.development_count)
                        status["sampled_development_jf_not_official"] = development_result["mean_jf"]
                        atomic_json(args.output / "STATUS.json", status)
            if world > 1 and (update % args.save_every == 0 or update == total_updates):
                dist.barrier()
            if update >= total_updates:
                stop = True
                break
        if stop:
            break
    if rank == 0:
        status.update({"status": "COMPLETE", "completed_updates": update, "elapsed_seconds": time.time() - started})
        atomic_json(args.output / "STATUS.json", status)
    if world > 1:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    root = Path("/9950backfile/chenjiahui/evo_artifacts")
    parser.add_argument("--foundation", type=Path, default=root / "models/Sa2VA-Qwen3-VL-4B-SAM3")
    parser.add_argument("--sam-checkpoint", type=Path, default=root / "models/SAM3.1-official-mirror/sam3.1_multiplex.pt")
    parser.add_argument("--sam-repo", type=Path, default=Path("/tmp/sam3_repo"))
    parser.add_argument("--long-root", type=Path, default=root / "datasets/long_rvos/train")
    parser.add_argument("--mevis-root", type=Path, default=root / "datasets/mevis_v2/train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--qwen-pixels", type=int, default=200704)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--grounding-lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--development-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.accumulation, args.frame_budget, args.save_every, args.epochs) < 1 or min(args.max_updates, args.development_count) < 0:
        parser.error("training counts must be positive (max-updates may be zero)")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except Exception as error:
        if int(os.environ.get("RANK", "0")) == 0:
            arguments.output.mkdir(parents=True, exist_ok=True)
            path = arguments.output / "STATUS.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            atomic_json(path, previous | {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        raise

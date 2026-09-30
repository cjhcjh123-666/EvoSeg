"""Train and evaluate the frozen-feature OPG SFT variants."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.factorized_temporal_spatial.protocol import load_mean_pool, temporal_inputs
from projects.evoseg.ordered_process_grounding.model import (
    OrderedResidualHead, TokenMeanPoolScorer, count_trainable_parameters,
)
from projects.evoseg.ordered_process_grounding.protocol import (
    block_swap_order,
    fixed_order_negative,
    is_order_sensitive,
    reverse_order,
)
from projects.evoseg.temporal_grounding_mechanism.common import (
    object_weighted,
    source_video_bootstrap,
    write_csv,
)
from projects.evoseg.temporal_grounding_mechanism.train_probe import examples


VARIANTS = {
    "opg_no_monotonic": {"alignment": "global", "lambda_order": 1.0},
    "opg_no_order_loss": {"alignment": "monotonic", "lambda_order": 0.0},
    "opg_full": {"alignment": "monotonic", "lambda_order": 1.0},
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def status(root: Path, state: str, phase: str, **values) -> None:
    record = {
        "state": state, "phase": phase, "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values,
    }
    atomic_json(root / "STATUS.json", record)
    (root / "PROGRESS.md").write_text(
        "# OPG SFT progress\n\n"
        f"- Updated: {record['updated_at']}\n- Phase: `{phase}`\n- State: `{state}`\n"
        + "".join(f"- {key}: {value}\n" for key, value in values.items())
    )


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def split_train(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    videos = sorted({row["video_id"] for row in rows})
    random.Random(42).shuffle(videos)
    validation_videos = set(videos[: max(8, len(videos) // 5)])
    return (
        [row for row in rows if row["video_id"] not in validation_videos],
        [row for row in rows if row["video_id"] in validation_videos],
    )


def base_logits(model, row: dict, device: str):
    import torch

    if isinstance(model, dict):
        model = model[row.get("_base_kind", "long_rvos")]
    with torch.no_grad():
        if row.get("_base_kind") == "groundmore":
            query, tracks = order_inputs(row, device)
            return model(query[0], tracks[0]).detach()
        query, states, tracks = temporal_inputs(row, device)
        return model(query, states, tracks).detach()


def order_supervised(row: dict) -> bool:
    return row.get("_base_kind") == "groundmore" or is_order_sensitive(row["expression"])


def order_inputs(row: dict, device: str, order: list[int] | None = None):
    import torch.nn.functional as functional

    tracks = functional.normalize(row["tracks"].float().to(device), dim=-1)
    if order is not None:
        tracks = tracks[:, order]
    query = row["query_tokens"].float().to(device)
    return query.unsqueeze(0), tracks.unsqueeze(0)


def losses(model, base_model, row: dict, device: str, lambda_order: float, margin: float):
    import torch
    import torch.nn.functional as functional

    base = base_logits(base_model, row, device).unsqueeze(0)
    query, tracks = order_inputs(row, device)
    final, ordered, audit = model(base, query, tracks)
    target = torch.tensor([row["target_index"]], device=device)
    identity_loss = functional.cross_entropy(final, target)
    order_loss = torch.zeros((), device=device)
    negative_kind = None
    if order_supervised(row):
        negative_kind, permutation = fixed_order_negative(row["identity"], tracks.shape[2])
        _, negative_tracks = order_inputs(row, device, permutation)
        negative, _ = model.order_scores(query, negative_tracks)
        target_index = row["target_index"]
        order_loss = functional.relu(
            margin - ordered[0, target_index] + negative[0, target_index]
        )
    return identity_loss + lambda_order * order_loss, {
        "identity": float(identity_loss.detach().cpu()),
        "order": float(order_loss.detach().cpu()),
        "alpha": float(audit["alpha"].detach().cpu()),
        "gate_mean": float(audit["gates"].mean().detach().cpu()),
        "negative_kind": negative_kind,
    }


def evaluate_loss(model, base_model, rows: list[dict], device: str, lambda_order: float, margin: float):
    import torch

    model.eval()
    identity = []
    order = []
    with torch.inference_mode():
        for row in rows:
            value, audit = losses(model, base_model, row, device, lambda_order, margin)
            identity.append(audit["identity"])
            if order_supervised(row):
                order.append(audit["order"])
    return float(np.mean(identity)), float(np.mean(order)) if order else 0.0


def train_one(
    fitting: list[dict], validation: list[dict], base_model, seed: int,
    variant: str, args,
):
    import torch

    config = VARIANTS[variant]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = OrderedResidualHead(alignment=config["alignment"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = random.Random(seed)
    history = []
    started = time.monotonic()
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    # Stage 1: exactly one identity-only warm-up epoch.
    for stage, epochs, order_weight in (
        ("identity_warmup", 1, 0.0),
        ("ordered_sft", args.epochs, config["lambda_order"]),
    ):
        best = None
        best_loss = float("inf")
        stale = 0
        for epoch in range(epochs):
            model.train()
            indices = list(range(len(fitting))); generator.shuffle(indices)
            totals = []; identities = []; orders = []
            for index in indices:
                row = fitting[index]
                total, audit = losses(model, base_model, row, args.device, order_weight, args.margin)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals.append(float(total.detach().cpu()))
                identities.append(audit["identity"]); orders.append(audit["order"])
            validation_identity, validation_order = evaluate_loss(
                model, base_model, validation, args.device, order_weight, args.margin
            )
            validation_total = validation_identity + order_weight * validation_order
            record = {
                "stage": stage, "epoch": epoch + 1,
                "train_total": float(np.mean(totals)),
                "train_identity": float(np.mean(identities)),
                "train_order": float(np.mean(orders)),
                "validation_total": validation_total,
                "validation_identity": validation_identity,
                "validation_order": validation_order,
                "alpha": float(torch.sigmoid(model.alpha).detach().cpu()),
            }
            history.append(record)
            if stage == "identity_warmup":
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                continue
            if validation_total < best_loss - 1e-5:
                best_loss = validation_total
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best)
    return model.eval(), {
        "seed": seed, "variant": variant, "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": count_trainable_parameters(model),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device))
        if str(args.device).startswith("cuda") else 0,
        "final_alpha": float(torch.sigmoid(model.alpha).detach().cpu()),
    }


def save_model(model, audit: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **{k: audit[k] for k in (
        "seed", "variant", "trainable_parameters", "final_alpha"
    )}}, path)
    path.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")


def load_model(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu")
    model = OrderedResidualHead(alignment=VARIANTS[payload["variant"]]["alignment"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def train_ground_base(fitting: list[dict], validation: list[dict], seed: int, args):
    import torch
    import torch.nn.functional as functional

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = TokenMeanPoolScorer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(seed); best=None; best_loss=float("inf"); stale=0; history=[]
    for epoch in range(args.epochs):
        model.train(); indices=list(range(len(fitting)));rng.shuffle(indices);train_losses=[]
        for index in indices:
            row=fitting[index];query,tracks=order_inputs(row,args.device);logits=model(query[0],tracks[0])
            target=torch.tensor([row["target_index"]],device=args.device);loss=functional.cross_entropy(logits[None],target)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval();values=[]
        with torch.inference_mode():
            for row in validation:
                query,tracks=order_inputs(row,args.device);logits=model(query[0],tracks[0]);target=torch.tensor([row["target_index"]],device=args.device)
                values.append(float(functional.cross_entropy(logits[None],target).cpu()))
        value=float(np.mean(values));history.append({"epoch":epoch+1,"train_loss":float(np.mean(train_losses)),"validation_loss":value})
        if value < best_loss - 1e-5:
            best_loss=value;best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=args.patience:break
    model.load_state_dict(best)
    return model.eval(), {"seed":seed,"history":history,"best_validation_loss":best_loss,"parameters":count_trainable_parameters(model)}


def save_ground_base(model, audit: dict, path: Path):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":model.state_dict(),"seed":audit["seed"]},path)
    path.with_suffix(".json").write_text(json.dumps(audit,indent=2)+"\n")


def load_ground_base(path: Path, device: str):
    import torch
    payload=torch.load(path,map_location="cpu");model=TokenMeanPoolScorer().to(device);model.load_state_dict(payload["state_dict"])
    return model.eval(),payload


def predict_variant(models, base_models, rows: list[dict], device: str):
    import torch

    predictions = {}
    order_scores = {}
    per_seed = {}
    started = time.monotonic()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for row in rows:
            if row["tracks"].shape[0] == 0:
                predictions[row["identity"]] = np.zeros(0, np.float32)
                order_scores[row["identity"]] = np.zeros(0, np.float32)
                continue
            query, tracks = order_inputs(row, device)
            finals = []; orders = []
            for model, base_model in zip(models, base_models):
                base = base_logits(base_model, row, device).unsqueeze(0)
                final, order, _ = model(base, query, tracks)
                finals.append(final[0].float().cpu().numpy())
                orders.append(order[0].float().cpu().numpy())
            predictions[row["identity"]] = np.mean(finals, axis=0)
            order_scores[row["identity"]] = np.mean(orders, axis=0)
            per_seed[row["identity"]] = finals
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return predictions, order_scores, per_seed, time.monotonic() - started, peak


def predict_mean(models, rows: list[dict], device: str):
    import torch

    result = {}; per_seed = {}
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            values = []
            if row["tracks"].shape[0]:
                values = [base_logits(model, row, device).float().cpu().numpy() for model in models]
                result[row["identity"]] = np.mean(values, axis=0)
            else:
                result[row["identity"]] = np.zeros(0, np.float32)
            per_seed[row["identity"]] = values
    return result, per_seed, time.monotonic() - started


def selected_row(row: dict, method: str, logits: np.ndarray, seed: str = "ensemble") -> dict:
    track_ids = [int(value) for value in row.get("candidate_track_ids", [])]
    selected_index = int(np.argmax(logits)) if len(logits) else None
    selected_id = track_ids[selected_index] if selected_index is not None else None
    return {
        "identity": row["identity"], "dataset": row["dataset"], "split": row["split"],
        "video_id": row["video_id"], "object_id": row["object_id"],
        "expression_id": row["expression_id"], "description_type": row["description_type"],
        "expression": row["expression"], "order_sensitive": int(is_order_sensitive(row["expression"])),
        "method": method, "seed": seed, "candidate_hit": int(row["candidate_hit"]),
        "candidate_count": len(track_ids), "oracle_track_id": row["oracle_track_id"],
        "selected_track_id": selected_id,
        "selection_correct": int(row["candidate_hit"] and selected_id == row["oracle_track_id"]),
        "J_and_F": float(row["candidate_jf"].get(selected_id, 0.0)),
    }


def summaries(rows: list[dict]) -> list[dict]:
    result = []
    for kind in ("static", "dynamic", "hybrid", "sequential"):
        for method in sorted({row["method"] for row in rows if row["seed"] == "ensemble"}):
            subset = [row for row in rows if row["seed"] == "ensemble" and row["description_type"] == kind and row["method"] == method]
            objects = object_weighted(subset, ["selection_correct", "J_and_F"])
            result.append({
                "description_type": kind, "method": method, "expressions": len(subset),
                "objects": len(objects), "candidate_hits": sum(row["candidate_hit"] for row in subset),
                "selection_accuracy": float(np.mean([row["selection_correct"] for row in objects])),
                "J_and_F": float(np.mean([row["J_and_F"] for row in objects])),
            })
    return result


def bootstrap(rows: list[dict], left: str, right: str, kind: str, field: str):
    grouped = defaultdict(dict); meta = {}
    for row in rows:
        if row["seed"] == "ensemble" and row["description_type"] == kind and row["method"] in {left, right}:
            grouped[row["identity"]][row["method"]] = float(row[field]); meta[row["identity"]] = row
    paired = []
    for identity, values in grouped.items():
        if len(values) == 2:
            paired.append({
                "video_id": meta[identity]["video_id"],
                "object_id": meta[identity]["object_id"],
                "description_type": kind,
                left: values[left], right: values[right],
            })
    return source_video_bootstrap(paired, left, right)


def order_diagnostics(model_sets: dict, base_models, rows: list[dict], device: str) -> list[dict]:
    import torch

    output = []
    with torch.inference_mode():
        for row in rows:
            if not order_supervised(row) or not row["candidate_hit"]:
                continue
            target = row["target_index"]
            query, original_tracks = order_inputs(row, device)
            for variant in ("opg_no_order_loss", "opg_full"):
                values = {}
                original_vectors = []
                for name, permutation in (
                    ("original", None), ("reverse", reverse_order(8)), ("block_swap", block_swap_order(8))
                ):
                    _, tracks = order_inputs(row, device, permutation)
                    vectors = [model.order_scores(query, tracks)[0][0].float().cpu().numpy() for model in model_sets[variant]]
                    vector = np.mean(vectors, axis=0)
                    values[name] = float(vector[target])
                    if name == "original": original_vectors = vector
                distractor = float(max(np.delete(original_vectors, target), default=values["original"]))
                output.append({
                    "identity": row["identity"], "dataset": row["dataset"], "video_id": row["video_id"],
                    "object_id": row["object_id"], "expression_id": row["expression_id"],
                    "description_type": row["description_type"], "expression": row["expression"],
                    "method": variant, "target_original_score": values["original"],
                    "target_reverse_score": values["reverse"], "target_block_swap_score": values["block_swap"],
                    "original_minus_reverse": values["original"] - values["reverse"],
                    "original_minus_block_swap": values["original"] - values["block_swap"],
                    "original_target_rank": int((-original_vectors).argsort().tolist().index(target) + 1),
                    "original_target_vs_distractor_margin": values["original"] - distractor,
                })
            # The frozen mean scorer is exactly permutation invariant by construction.
            base_values = []
            for base in base_models:
                base_values.append(base_logits(base, row, device)[target].item())
            base_value = float(np.mean(base_values))
            output.append({
                "identity": row["identity"], "dataset": row["dataset"], "video_id": row["video_id"],
                "object_id": row["object_id"], "expression_id": row["expression_id"],
                "description_type": row["description_type"], "expression": row["expression"],
                "method": "mean_pool", "target_original_score": base_value,
                "target_reverse_score": base_value, "target_block_swap_score": base_value,
                "original_minus_reverse": 0.0, "original_minus_block_swap": 0.0,
                "original_target_rank": None,
                "original_target_vs_distractor_margin": None,
            })
    return output


def run(args) -> None:
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    status(root, "running", "load")
    train_all, train_hits = examples(Path(args.train_features), Path(args.train_metrics), args.hit_threshold)
    evaluation, _ = examples(Path(args.eval_features), Path(args.eval_metrics), args.hit_threshold)
    for row in train_all + evaluation: row["_base_kind"] = "long_rvos"
    if {row["video_id"] for row in train_all} & {row["video_id"] for row in evaluation}:
        raise RuntimeError("train/evaluation source-video overlap")
    for row in train_all + evaluation:
        if row["tracks"].shape[1] != 8 or "query_tokens" not in row:
            raise RuntimeError(f"OPG requires frozen T=8 tracks and query tokens: {row['identity']}")
    fitting, validation = split_train(train_hits)
    ground_train_all=[];ground_train_hits=[];ground_evaluation=[];ground_fitting=[];ground_validation=[]
    if args.ground_train_features:
        ground_train_all,ground_train_hits=examples(Path(args.ground_train_features),Path(args.ground_train_metrics),args.hit_threshold)
        ground_evaluation,_=examples(Path(args.ground_eval_features),Path(args.ground_eval_metrics),args.hit_threshold)
        for row in ground_train_all+ground_evaluation: row["_base_kind"]="groundmore"
        if {row["video_id"] for row in ground_train_all} & {row["video_id"] for row in ground_evaluation}:
            raise RuntimeError("GroundMoRe train/test source-video overlap")
        ground_fitting,ground_validation=split_train(ground_train_hits)
    long_fitting,long_validation=fitting,validation
    fitting=long_fitting+ground_fitting;validation=long_validation+ground_validation
    order_train = [row for row in fitting if order_supervised(row)]
    order_validation = [row for row in validation if order_supervised(row)]

    long_base_models = []
    for seed in args.seeds:
        model, payload = load_mean_pool(Path(args.mean_checkpoints) / f"mean_pool_seed{seed}.pt", args.device)
        if payload["seed"] != seed:
            raise RuntimeError("mean-pool checkpoint seed mismatch")
        for parameter in model.parameters(): parameter.requires_grad_(False)
        long_base_models.append(model)
    ground_base_models=[];ground_base_audits=[]
    if ground_fitting:
        status(root,"running","train_groundmore_mean_pool",seeds=args.seeds)
        for seed in args.seeds:
            path=root/"checkpoints"/f"ground_mean_pool_seed{seed}.pt"
            if path.is_file(): model,_=load_ground_base(path,args.device);audit=json.loads(path.with_suffix('.json').read_text())
            else:
                model,audit=train_ground_base(ground_fitting,ground_validation,seed,args);save_ground_base(model,audit,path)
            for parameter in model.parameters():parameter.requires_grad_(False)
            ground_base_models.append(model);ground_base_audits.append(audit)
    base_models=[
        {"long_rvos":long_model,"groundmore":ground_model} if ground_base_models else long_model
        for long_model,ground_model in zip(long_base_models,ground_base_models or long_base_models)
    ]

    model_sets = {variant: [] for variant in VARIANTS}
    training = []
    planned = len(VARIANTS) * len(args.seeds)
    for variant in VARIANTS:
        for index, (seed, base) in enumerate(zip(args.seeds, base_models)):
            path = root / "checkpoints" / f"{variant}_seed{seed}.pt"
            if path.is_file():
                model, _ = load_model(path, args.device)
                audit = json.loads(path.with_suffix(".json").read_text())
            else:
                model, audit = train_one(fitting, validation, base, seed, variant, args)
                save_model(model, audit, path)
            model_sets[variant].append(model); training.append(audit)
            status(root, "running", "training", completed=len(training), planned=planned, variant=variant, seed=seed)

    status(root, "running", "long_rvos_evaluation", expressions=len(evaluation))
    mean_predictions, mean_seeds, mean_latency = predict_mean(base_models, evaluation, args.device)
    variant_outputs = {}
    runtime = [{"method": "mean_pool", "latency_seconds": mean_latency, "expressions": len(evaluation)}]
    for variant, models in model_sets.items():
        pred, order, seed_values, latency, peak = predict_variant(models, base_models, evaluation, args.device)
        variant_outputs[variant] = (pred, order, seed_values)
        runtime.append({"method": variant, "latency_seconds": latency, "expressions": len(evaluation), "peak_memory_bytes": peak})

    rows = []
    for row in evaluation:
        rows.append(selected_row(row, "mean_pool", mean_predictions[row["identity"]]))
        for seed, values in zip(args.seeds, mean_seeds[row["identity"]]):
            rows.append(selected_row(row, "mean_pool", values, str(seed)))
        for variant, (prediction, _order, seed_values) in variant_outputs.items():
            rows.append(selected_row(row, variant, prediction[row["identity"]]))
            for seed, values in zip(args.seeds, seed_values.get(row["identity"], [])):
                rows.append(selected_row(row, variant, values, str(seed)))
    write_csv(root / "per_expression.csv", rows)
    summary = summaries(rows)
    write_csv(root / "long_rvos_summary.csv", summary)
    order_rows = order_diagnostics(model_sets, base_models, evaluation, args.device)
    write_csv(root / "order_pairs.csv", order_rows)
    write_csv(root / "runtime.csv", runtime)

    ground_rows=[];ground_summary=[];ground_comparison=None;ground_order_rows=[]
    if ground_evaluation:
        status(root,"running","groundmore_evaluation",expressions=len(ground_evaluation))
        ground_mean,ground_mean_seeds,ground_mean_latency=predict_mean(base_models,ground_evaluation,args.device)
        ground_outputs={}
        for variant,models in model_sets.items():
            ground_outputs[variant]=predict_variant(models,base_models,ground_evaluation,args.device)
        for row in ground_evaluation:
            ground_rows.append(selected_row(row,"mean_pool",ground_mean[row["identity"]]))
            for seed,values in zip(args.seeds,ground_mean_seeds[row["identity"]]):ground_rows.append(selected_row(row,"mean_pool",values,str(seed)))
            for variant,(prediction,_orders,seeds_values,_latency,_peak) in ground_outputs.items():
                ground_rows.append(selected_row(row,variant,prediction[row["identity"]]))
                for seed,values in zip(args.seeds,seeds_values.get(row["identity"],[])):ground_rows.append(selected_row(row,variant,values,str(seed)))
        write_csv(root/"groundmore_sequential.csv",ground_rows)
        ground_summary=summaries(ground_rows)
        ground_comparison={
            "J_and_F":bootstrap(ground_rows,"opg_full","mean_pool","sequential","J_and_F"),
            "selection_accuracy":bootstrap(ground_rows,"opg_full","mean_pool","sequential","selection_correct"),
        }
        ground_order_rows=order_diagnostics(model_sets,base_models,ground_evaluation,args.device)
        write_csv(root/"groundmore_order_pairs.csv",ground_order_rows)

    comparisons = {}
    for kind in ("static", "dynamic", "hybrid"):
        comparisons[kind] = {
            "J_and_F": bootstrap(rows, "opg_full", "mean_pool", kind, "J_and_F"),
            "selection_accuracy": bootstrap(rows, "opg_full", "mean_pool", kind, "selection_correct"),
        }
    order_summary = {}
    for method in ("mean_pool", "opg_no_order_loss", "opg_full"):
        subset = [row for row in order_rows if row["method"] == method]
        order_summary[method] = {
            "samples": len(subset),
            "original_minus_reverse": source_video_bootstrap(subset, "target_original_score", "target_reverse_score"),
            "original_minus_block_swap": source_video_bootstrap(subset, "target_original_score", "target_block_swap_score"),
            "reverse_sensitivity_rate": float(np.mean([row["original_minus_reverse"] > 0 for row in subset])) if subset else None,
            "block_swap_sensitivity_rate": float(np.mean([row["original_minus_block_swap"] > 0 for row in subset])) if subset else None,
        }
    audit = {
        "train_expressions": len(train_all), "train_candidate_hits": len(train_hits),
        "fit_expressions": len(fitting), "early_stop_expressions": len(validation),
        "order_fit_expressions": len(order_train), "order_fit_videos": len({row['video_id'] for row in order_train}),
        "order_early_stop_expressions": len(order_validation),
        "evaluation_expressions": len(evaluation),
        "evaluation_order_expressions": sum(order_supervised(row) for row in evaluation),
        "groundmore_train_expressions":len(ground_train_all),"groundmore_train_candidate_hits":len(ground_train_hits),
        "groundmore_eval_expressions":len(ground_evaluation),"groundmore_summary":ground_summary,
        "groundmore_comparison":ground_comparison,"groundmore_base_training":ground_base_audits,
        "ground_truth_used_for_inference": False,
        "trainable_parameters": count_trainable_parameters(model_sets["opg_full"][0]),
        "margin": args.margin, "lambda_order": 1.0, "seeds": args.seeds,
        "training": training, "summary": summary, "comparisons": comparisons,
        "order_summary": order_summary, "runtime": runtime,
        "inputs": {
            "train_features": str(Path(args.train_features).resolve()),
            "train_metrics": str(Path(args.train_metrics).resolve()),
            "eval_features": str(Path(args.eval_features).resolve()),
            "eval_metrics": str(Path(args.eval_metrics).resolve()),
        },
    }
    atomic_json(root / "summary.json", audit)
    status(root, "complete", "evaluation_complete", expressions=len(evaluation), groundmore_expressions=len(ground_evaluation))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True); parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--eval-features", required=True); parser.add_argument("--eval-metrics", required=True)
    parser.add_argument("--mean-checkpoints", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--ground-train-features"); parser.add_argument("--ground-train-metrics")
    parser.add_argument("--ground-eval-features"); parser.add_argument("--ground-eval-metrics")
    parser.add_argument("--device", default="cuda:1"); parser.add_argument("--seeds", type=int, nargs="+", default=[11,23,42])
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6); parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

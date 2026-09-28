"""Train parameter-identical static and temporal diagnostic BiGRU probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np

from projects.evoseg.temporal_grounding_mechanism.common import (
    source_video_bootstrap,
    write_csv,
)


class ProbeFactory:
    @staticmethod
    def build(query_dim: int = 256, visual_dim: int = 1152, hidden: int = 64):
        import torch.nn as nn

        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(query_dim, hidden * 2)
                self.state = nn.Linear(query_dim, hidden * 2)
                self.visual = nn.Linear(visual_dim, hidden * 2)
                self.gru = nn.GRU(hidden * 2, hidden, batch_first=True, bidirectional=True)
                self.score = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 1))

            def forward(self, query, states, tracks):
                # query [C,Q], states [C,T,Q], tracks [C,T,V]
                values = self.visual(tracks) + self.state(states) + self.query(query).unsqueeze(1)
                encoded, _ = self.gru(values)
                return self.score(encoded[:, -1]).squeeze(-1)

        return Probe()


def load_features(path: Path) -> tuple[list[dict], dict[str, dict]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    metadata = json.loads(path.with_suffix(".json").read_text())
    return payload, {row["identity"]: row for row in metadata["records"]}


def load_metrics(path: Path) -> dict[str, dict]:
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["prompt_method"] == "concept":
                key = "/".join((row["dataset"], row["video_id"], row["object_id"], row["expression_id"]))
                row["candidate_metrics"] = json.loads(row["candidate_metrics"])
                result[key] = row
    return result


def examples(feature_path: Path, metric_path: Path, hit_threshold: float) -> tuple[list[dict], list[dict]]:
    payload, metadata = load_features(feature_path)
    metrics = load_metrics(metric_path)
    all_rows = []
    trainable = []
    for value in payload:
        key = value["identity"]
        meta = metadata[key]
        track_ids = [int(value) for value in meta["candidate_track_ids"]]
        metric = metrics.get(key)
        if metric is None:
            if track_ids or not meta.get("candidate_generation_failure"):
                raise RuntimeError(f"missing candidate metrics for non-failure expression: {key}")
            oracle_id = None
            hit = False
            candidate_jf = {}
        else:
            oracle_id = int(metric["oracle_track_id"]) if metric["oracle_track_id"] else None
            hit = oracle_id in track_ids and float(metric["oracle_J_and_F"]) >= hit_threshold
            candidate_jf = {
                int(row["track_id"]): float(row["J_and_F"])
                for row in metric["candidate_metrics"]
            }
        row = {
            **value,
            **meta,
            "oracle_track_id": oracle_id,
            "candidate_hit": hit,
            "candidate_jf": candidate_jf,
            "target_index": track_ids.index(oracle_id) if hit else None,
        }
        all_rows.append(row)
        if hit:
            trainable.append(row)
    return all_rows, trainable


def model_inputs(example: dict, temporal: bool, device: str):
    import torch.nn.functional as functional

    tracks = functional.normalize(example["tracks"].float().to(device), dim=-1)
    if temporal:
        states = functional.normalize(example["temporal_states"].float().to(device), dim=-1)
        states = states.unsqueeze(0).expand(tracks.shape[0], -1, -1)
        query = states[:, -1]
    else:
        tracks = tracks[:, -1:].expand(-1, 4, -1)
        query_vector = functional.normalize(example["static_query"].float().to(device), dim=-1)
        states = query_vector.view(1, 1, -1).expand(tracks.shape[0], 4, -1)
        query = query_vector.unsqueeze(0).expand(tracks.shape[0], -1)
    return query, states, tracks


def train_one(train, validation, temporal: bool, seed: int, args):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ProbeFactory.build().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    stale = 0
    history = []
    rng = random.Random(seed)
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for index in order:
            example = train[index]
            query, states, tracks = model_inputs(example, temporal, args.device)
            logits = model(query, states, tracks)
            target = torch.tensor([example["target_index"]], device=args.device)
            loss = functional.cross_entropy(logits[None], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        values = []
        with torch.inference_mode():
            for example in validation:
                query, states, tracks = model_inputs(example, temporal, args.device)
                logits = model(query, states, tracks)
                target = torch.tensor([example["target_index"]], device=args.device)
                values.append(float(functional.cross_entropy(logits[None], target).cpu()))
        validation_loss = float(np.mean(values))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best)
    return model.eval(), history


def predict(models, rows, temporal: bool, device: str) -> dict[str, np.ndarray]:
    import torch

    result = {}
    with torch.inference_mode():
        for row in rows:
            if row["tracks"].shape[0] == 0:
                result[row["identity"]] = np.zeros(0, dtype=np.float32)
                continue
            values = []
            for model in models:
                query, states, tracks = model_inputs(row, temporal, device)
                values.append(model(query, states, tracks).float().cpu().numpy())
            result[row["identity"]] = np.mean(values, axis=0)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_models(models, histories, args, fitting, validation) -> dict:
    import torch

    root = Path(args.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for temporal, name in ((False, "static"), (True, "temporal")):
        for seed, model in zip(args.seeds, models[temporal]):
            path = root / f"{name}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": name,
                    "seed": seed,
                    "architecture": "parameter-identical single-layer BiGRU",
                },
                path,
            )
            files[path.name] = sha256(path)
    manifest = {
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "hit_threshold": args.hit_threshold,
        "train_features": str(Path(args.train_features).resolve()),
        "train_metrics": str(Path(args.train_metrics).resolve()),
        "fit_identities": [row["identity"] for row in fitting],
        "early_stop_identities": [row["identity"] for row in validation],
        "files": files,
        "histories": histories,
    }
    (root / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_models(args) -> tuple[dict, dict, dict]:
    import torch

    root = Path(args.load_checkpoint_dir)
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    if manifest["seeds"] != args.seeds:
        raise RuntimeError("checkpoint seeds differ from requested preregistered seeds")
    for name in ("epochs", "patience", "learning_rate", "hit_threshold"):
        if manifest[name] != getattr(args, name):
            raise RuntimeError(f"checkpoint {name} differs from evaluation argument")
    models = {False: [], True: []}
    for temporal, name in ((False, "static"), (True, "temporal")):
        for seed in args.seeds:
            path = root / f"{name}_seed{seed}.pt"
            if sha256(path) != manifest["files"][path.name]:
                raise RuntimeError(f"checkpoint hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu")
            if payload["method"] != name or payload["seed"] != seed:
                raise RuntimeError(f"checkpoint metadata mismatch: {path}")
            model = ProbeFactory.build().to(args.device)
            model.load_state_dict(payload["state_dict"])
            models[temporal].append(model.eval())
    return models, manifest.get("histories", {}), manifest


def build_result_rows(evaluation: list[dict], predictions: dict) -> list[dict]:
    rows = []
    for example in evaluation:
        value = {
            key: example[key]
            for key in (
                "identity",
                "dataset",
                "split",
                "video_id",
                "object_id",
                "expression_id",
                "description_type",
                "expression",
            )
        }
        value["candidate_hit"] = int(example["candidate_hit"])
        value["oracle_track_id"] = example["oracle_track_id"]
        track_ids = [int(item) for item in example["candidate_track_ids"]]
        for method in ("static", "temporal"):
            logits = predictions[method][example["identity"]]
            selected_index = int(np.argmax(logits)) if len(logits) else None
            selected_id = track_ids[selected_index] if selected_index is not None else None
            value[f"{method}_selected_track_id"] = selected_id
            value[f"{method}_selection_correct"] = int(
                example["candidate_hit"] and selected_id == example["oracle_track_id"]
            )
            value[f"{method}_candidate_direct_J_and_F"] = example["candidate_jf"].get(
                selected_id, 0.0
            )
            value[f"{method}_target_margin"] = (
                float(
                    logits[example["target_index"]]
                    - max(np.delete(logits, example["target_index"]), default=0.0)
                )
                if example["candidate_hit"] and len(logits)
                else None
            )
        rows.append(value)
    return rows


def run(args) -> None:
    import torch

    train_all, train_hits = examples(Path(args.train_features), Path(args.train_metrics), args.hit_threshold)
    evaluation, _ = examples(Path(args.eval_features), Path(args.eval_metrics), args.hit_threshold)
    train_videos = sorted({row["video_id"] for row in train_hits})
    eval_videos = {row["video_id"] for row in evaluation}
    if set(train_videos) & eval_videos:
        raise RuntimeError("official train/evaluation source-video overlap")
    rng = random.Random(42)
    rng.shuffle(train_videos)
    validation_videos = set(train_videos[: max(8, len(train_videos) // 5)])
    fitting = [row for row in train_hits if row["video_id"] not in validation_videos]
    validation = [row for row in train_hits if row["video_id"] in validation_videos]
    checkpoint_manifest = None
    if args.load_checkpoint_dir:
        models, histories, checkpoint_manifest = load_models(args)
        if checkpoint_manifest["fit_identities"] != [row["identity"] for row in fitting]:
            raise RuntimeError("checkpoint fitting split differs from reconstructed official-train split")
        if checkpoint_manifest["early_stop_identities"] != [row["identity"] for row in validation]:
            raise RuntimeError("checkpoint early-stop split differs from reconstructed official-train split")
    else:
        models = {False: [], True: []}
        histories = {}
        for temporal in (False, True):
            for seed in args.seeds:
                model, history = train_one(fitting, validation, temporal, seed, args)
                models[temporal].append(model)
                histories[f"{'temporal' if temporal else 'static'}_{seed}"] = history
        if args.checkpoint_dir:
            checkpoint_manifest = save_models(models, histories, args, fitting, validation)
    predictions = {
        "static": predict(models[False], evaluation, False, args.device),
        "temporal": predict(models[True], evaluation, True, args.device),
    }
    rows = build_result_rows(evaluation, predictions)
    dynamic_hits = [row for row in rows if row["description_type"] == "dynamic" and row["candidate_hit"]]
    bootstrap_accuracy = source_video_bootstrap(dynamic_hits, "temporal_selection_correct", "static_selection_correct")
    bootstrap_jf = source_video_bootstrap(
        [row for row in rows if row["description_type"] == "dynamic"],
        "temporal_candidate_direct_J_and_F",
        "static_candidate_direct_J_and_F",
    )
    seed_dynamic_results = []
    detailed_rows = [{**row, "model_seed": "ensemble"} for row in rows]
    for index, seed in enumerate(args.seeds):
        seed_rows = build_result_rows(
            evaluation,
            {
                "static": predict([models[False][index]], evaluation, False, args.device),
                "temporal": predict([models[True][index]], evaluation, True, args.device),
            },
        )
        seed_dynamic_hits = [
            row
            for row in seed_rows
            if row["description_type"] == "dynamic" and row["candidate_hit"]
        ]
        seed_dynamic_all = [
            row for row in seed_rows if row["description_type"] == "dynamic"
        ]
        seed_dynamic_results.append(
            {
                "seed": seed,
                "selection_accuracy": source_video_bootstrap(
                    seed_dynamic_hits,
                    "temporal_selection_correct",
                    "static_selection_correct",
                ),
                "candidate_direct_J_and_F": source_video_bootstrap(
                    seed_dynamic_all,
                    "temporal_candidate_direct_J_and_F",
                    "static_candidate_direct_J_and_F",
                ),
            }
        )
        detailed_rows.extend({**row, "model_seed": seed} for row in seed_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "probe_results.csv", rows)
    write_csv(output / "probe_results_by_model.csv", detailed_rows)
    (output / "probe_audit.json").write_text(
        json.dumps(
            {
                "command": [sys.executable, *sys.argv],
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name(args.device),
                },
                "inputs": {
                    "train_features": str(Path(args.train_features).resolve()),
                    "train_metrics": str(Path(args.train_metrics).resolve()),
                    "eval_features": str(Path(args.eval_features).resolve()),
                    "eval_metrics": str(Path(args.eval_metrics).resolve()),
                },
                "official_train_feature_records": len(train_all),
                "official_train_candidate_hits": len(train_hits),
                "fit_expressions": len(fitting),
                "validation_expressions": len(validation),
                "evaluation_expressions": len(evaluation),
                "train_eval_video_overlap": [],
                "candidate_miss_definition": f"oracle J&F < {args.hit_threshold} or no oracle candidate",
                "architecture": "parameter-identical single-layer BiGRU; static repeats final anchor/state four times; temporal reads four ordered region/state pairs",
                "trainable_parameters": sum(parameter.numel() for parameter in models[False][0].parameters()),
                "seeds": args.seeds,
                "checkpoint_mode": "load_only" if args.load_checkpoint_dir else "trained_on_official_train",
                "checkpoint_dir": str(
                    Path(args.load_checkpoint_dir or args.checkpoint_dir).resolve()
                ) if (args.load_checkpoint_dir or args.checkpoint_dir) else None,
                "checkpoint_manifest": checkpoint_manifest,
                "dynamic_bootstrap_selection_accuracy": bootstrap_accuracy,
                "dynamic_bootstrap_candidate_direct_J_and_F": bootstrap_jf,
                "per_seed_dynamic_results": seed_dynamic_results,
                "histories": histories,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"dynamic_accuracy": bootstrap_accuracy, "dynamic_J_and_F": bootstrap_jf}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--eval-features", required=True)
    parser.add_argument("--eval-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-dir")
    checkpoint.add_argument("--load-checkpoint-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

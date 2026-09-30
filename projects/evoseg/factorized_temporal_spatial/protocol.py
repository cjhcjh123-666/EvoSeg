"""Frozen-feature identity-selector protocol for FTSG."""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np


class MeanPoolProbeFactory:
    """Order-free control within 0.2% of the BiGRU parameter count."""

    @staticmethod
    def build(query_dim: int = 256, visual_dim: int = 1152, hidden: int = 160):
        import torch.nn as nn

        class MeanPoolProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(query_dim, hidden)
                self.state = nn.Linear(query_dim, hidden)
                self.visual = nn.Linear(visual_dim, hidden)
                self.score = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                )

            def forward(self, query, states, tracks):
                pooled = (
                    self.query(query)
                    + self.state(states).mean(dim=1)
                    + self.visual(tracks).mean(dim=1)
                )
                return self.score(pooled).squeeze(-1)

        return MeanPoolProbe()


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def temporal_inputs(example: dict, device: str, track_order: list[int] | None = None):
    import torch.nn.functional as functional

    tracks = functional.normalize(example["tracks"].float().to(device), dim=-1)
    if track_order is not None:
        tracks = tracks[:, track_order]
    states = functional.normalize(example["temporal_states"].float().to(device), dim=-1)
    states = states.unsqueeze(0).expand(tracks.shape[0], -1, -1)
    query = states[:, -1]
    return query, states, tracks


def fixed_derangement(identity: str, length: int = 4, seed: int = 42) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    original = list(range(length))
    value = original[:]
    while value == original:
        rng.shuffle(value)
    return value


def predict(models, rows: list[dict], device: str, mode: str) -> tuple[dict[str, np.ndarray], float, int]:
    import torch

    result = {}
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            if row["tracks"].shape[0] == 0:
                result[row["identity"]] = np.zeros(0, dtype=np.float32)
                continue
            order = fixed_derangement(row["identity"]) if mode == "order_shuffle" else None
            query, states, tracks = temporal_inputs(row, device, order)
            values = [model(query, states, tracks).float().cpu().numpy() for model in models]
            result[row["identity"]] = np.mean(values, axis=0)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return result, time.monotonic() - started, peak


def train_mean_pool(
    fitting: list[dict],
    validation: list[dict],
    seed: int,
    device: str,
    epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 3e-4,
):
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MeanPoolProbeFactory.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    best_epoch = None
    stale = 0
    history = []
    rng = random.Random(seed)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(fitting)))
        rng.shuffle(order)
        losses = []
        for index in order:
            row = fitting[index]
            query, states, tracks = temporal_inputs(row, device)
            logits = model(query, states, tracks)
            target = torch.tensor([row["target_index"]], device=device)
            loss = functional.cross_entropy(logits[None], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        losses_validation = []
        with torch.inference_mode():
            for row in validation:
                query, states, tracks = temporal_inputs(row, device)
                logits = model(query, states, tracks)
                target = torch.tensor([row["target_index"]], device=device)
                losses_validation.append(float(functional.cross_entropy(logits[None], target).cpu()))
        validation_loss = float(np.mean(losses_validation))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best)
    peak = int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0
    return model.eval(), {
        "seed": seed,
        "state": "complete",
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": peak,
        "parameter_count": count_parameters(model),
    }


def save_mean_pool(model, status: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mean-pool order-free control",
            "seed": status["seed"],
            "state_dict": model.state_dict(),
            "parameter_count": status["parameter_count"],
        },
        path,
    )


def load_mean_pool(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu")
    model = MeanPoolProbeFactory.build().to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload

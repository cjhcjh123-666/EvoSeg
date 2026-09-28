from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def identity(dataset: str, video: str, object_id: str, expression_id: str) -> str:
    return "/".join(map(str, (dataset, video, object_id, expression_id)))


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def object_weighted(rows: list[dict], value_fields: list[str]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["object_id"], row["description_type"])].append(row)
    result = []
    for (video, object_id, description_type), values in sorted(grouped.items()):
        result.append(
            {
                "video_id": video,
                "object_id": object_id,
                "description_type": description_type,
                **{
                    field: float(np.mean([float(value[field]) for value in values]))
                    for field in value_fields
                },
            }
        )
    return result


def source_video_bootstrap(
    rows: list[dict], left: str, right: str, iterations: int = 2000, seed: int = 42
) -> dict:
    """Object-weight first, then resample source videos with replacement."""
    objects = object_weighted(rows, [left, right])
    by_video: dict[str, list[float]] = defaultdict(list)
    for row in objects:
        by_video[row["video_id"]].append(float(row[left]) - float(row[right]))
    videos = sorted(by_video)
    if not videos:
        return {"mean": None, "ci_low": None, "ci_high": None, "objects": 0, "videos": 0}
    observed = [value for values in by_video.values() for value in values]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        chosen = rng.choice(videos, len(videos), replace=True)
        samples.append(float(np.mean([x for video in chosen for x in by_video[video]])))
    return {
        "mean": float(np.mean(observed)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "objects": len(objects),
        "videos": len(videos),
        "iterations": iterations,
        "unit": "source_video",
    }


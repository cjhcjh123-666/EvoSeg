"""Correct an already finished capability gate without rewriting raw results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(run_dir: Path, minimum: float) -> dict:
    records = [json.loads(line) for line in (run_dir / "training.jsonl").read_text().splitlines()]
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_epoch[int(record["epoch"])].append(record)
    epochs = sorted(by_epoch)
    means = {}
    for epoch in epochs:
        group = by_epoch[epoch]
        mask = sum(item["loss_bce"] + item["loss_dice"] for item in group) / len(group)
        means[str(epoch)] = {
            "samples": len(group),
            "total_loss": sum(item["loss"] for item in group) / len(group),
            "mask_bce_dice": mask,
            "selection_ce": sum(item["loss_selection"] for item in group) / len(group),
        }
    first = means[str(epochs[0])]["mask_bce_dice"]
    last = means[str(epochs[-1])]["mask_bce_dice"]
    improvement = (first - last) / first
    payload = {
        "status": "PASS" if improvement >= minimum else "FAIL",
        "gate": "QwenSeg-SAM31 mask-learning capability (corrected)",
        "raw_result_preserved": str(run_dir / "result.json"),
        "correction_reason": (
            "The raw gate incorrectly applied the threshold to total loss including "
            "auxiliary query-selection CE; the preregistered capability criterion is mask BCE+Dice."
        ),
        "required_relative_mask_improvement": minimum,
        "first_epoch_mask_loss": first,
        "last_epoch_mask_loss": last,
        "relative_mask_improvement": improvement,
        "query_ranking_stability": "UNRESOLVED_REQUIRES_VALIDATION_PILOT",
        "epoch_means": means,
    }
    (run_dir / "corrected_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(main(args.run_dir, args.minimum), indent=2))

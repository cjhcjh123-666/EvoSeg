"""Select source videos before objects for the fixed ProcessVIRST pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def select_videos(manifest: dict, count: int, seed: int = 42) -> dict:
    video_ids = sorted({item["video_id"] for item in manifest["objects"]})
    selected = set(random.Random(seed).sample(video_ids, min(count, len(video_ids))))
    objects = [item for item in manifest["objects"] if item["video_id"] in selected]
    return {
        **manifest,
        "objects": objects,
        "process_virst_pilot": {
            "sampling_unit": "source_video_before_object",
            "seed": seed,
            "requested_videos": count,
            "available_videos": len(video_ids),
            "selected_videos": sorted(selected),
            "selected_video_count": len(selected),
            "selected_object_count": len(objects),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    value = select_videos(json.loads(args.manifest.read_text()), args.videos, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(value["process_virst_pilot"], ensure_ascii=False))


if __name__ == "__main__":
    main()

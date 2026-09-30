"""Create a deterministic two-source-video Long-RVOS smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=2)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text())
    selected_videos = []
    selected_objects = []
    for item in value["objects"]:
        video_id = item["video_id"]
        if video_id not in selected_videos:
            if len(selected_videos) >= args.videos:
                continue
            selected_videos.append(video_id)
        if video_id in selected_videos:
            selected_objects.append(item)
    value["objects"] = selected_objects
    value["selection"] = {
        **value.get("selection", {}),
        "process_virst_smoke": True,
        "source_videos": selected_videos,
        "objects": len(selected_objects),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"videos": selected_videos, "objects": len(selected_objects)}))


if __name__ == "__main__":
    main()

"""Compose paper-ready Frame Prompt versus FTG qualitative comparisons."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw


def _expression_lookup(manifest: Path) -> dict[tuple[str, str, str], str]:
    payload = json.loads(manifest.read_text())
    return {
        (record["dataset"], str(record["video_id"]), str(record["expression_id"])):
        record["expression"]
        for record in payload["records"]
    }


def _sample_key(filename: str) -> tuple[str, str, str] | None:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    # Contact sheets are index_dataset_expression-type_video-expression.  The
    # two public dataset names are the only component containing an underscore.
    dataset = "long_rvos" if parts[1:3] == ["long", "rvos"] else "mevis_v2"
    offset = 3 if dataset == "long_rvos" else 3
    return dataset, parts[offset + 1], parts[offset + 2]


def compose_pair(
    frame_image: Path,
    ftg_image: Path,
    output: Path,
    expression: str = "",
) -> None:
    frame = Image.open(frame_image).convert("RGB")
    ftg = Image.open(ftg_image).convert("RGB")
    if frame.size != ftg.size:
        raise ValueError("paired contact sheets must have equal dimensions")
    title_lines = textwrap.wrap(expression, width=105) if expression else []
    header = 52 + 18 * len(title_lines)
    row_label = 24
    canvas = Image.new(
        "RGB", (frame.width, header + 2 * (row_label + frame.height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Yellow: overlap   Green: missed GT   Red: false positive", fill="black")
    for index, line in enumerate(title_lines):
        draw.text((8, 29 + 18 * index), f"Query: {line}" if index == 0 else line, fill="black")
    y = header
    draw.rectangle((0, y, frame.width, y + row_label), fill=(235, 235, 235))
    draw.text((8, y + 5), "Frame Prompt", fill="black")
    canvas.paste(frame, (0, y + row_label))
    y += row_label + frame.height
    draw.rectangle((0, y, frame.width, y + row_label), fill=(225, 235, 255))
    draw.text((8, y + 5), "FTG", fill="black")
    canvas.paste(ftg, (0, y + row_label))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def compose_directory(
    frame_dir: Path,
    ftg_dir: Path,
    manifest: Path,
    output: Path,
) -> list[Path]:
    lookup = _expression_lookup(manifest)
    written = []
    for frame_image in sorted(frame_dir.glob("*.jpg")):
        ftg_image = ftg_dir / frame_image.name
        if not ftg_image.is_file():
            continue
        key = _sample_key(frame_image.name)
        expression = lookup.get(key, "") if key else ""
        destination = output / frame_image.name
        compose_pair(frame_image, ftg_image, destination, expression)
        written.append(destination)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--ftg-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = compose_directory(args.frame_dir, args.ftg_dir, args.manifest, args.output)
    print(json.dumps({"count": len(paths), "outputs": [str(path) for path in paths]}, indent=2))

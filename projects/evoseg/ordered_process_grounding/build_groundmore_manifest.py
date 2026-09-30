"""Build a Sequential-only GroundMoRe manifest from official metadata/files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def time_str_to_seconds(value: str) -> int:
    """Exact time conversion used by the official GroundMoRe evaluator."""
    parts=[int(part) for part in value.split(":")]
    if len(parts)==2:return parts[0]*60+parts[1]
    if len(parts)==3:return parts[0]*3600+parts[1]*60+parts[2]
    raise ValueError(f"unsupported official GroundMoRe time: {value}")


def official_action_window(video_id: str, question: dict) -> tuple[float,float]:
    clip_code=video_id[-9:].split("_")[0]
    clip_start=time_str_to_seconds(clip_code[:2]+":"+clip_code[2:])
    return (
        float((time_str_to_seconds(question["action_start"])-clip_start)*6),
        float((time_str_to_seconds(question["action_end"])-clip_start)*6-1),
    )


def run(args) -> None:
    metadata_path = Path(args.metadata).resolve()
    videos = json.loads(metadata_path.read_text())["videos"]
    data_root = Path(args.data_root).resolve()
    image_root = data_root / "annotations"
    binary_root = Path(args.binary_mask_root).resolve() / args.split
    objects=[];failures=[];official_sequential_expressions=0
    for video_id,video in sorted(videos.items()):
        sequential=[(str(expression_id),question) for expression_id,question in video["questions"].items() if question["q_type"].lower()=="sequential"]
        if not sequential:continue
        official_sequential_expressions+=len(sequential)
        directory = image_root / video_id
        frames=sorted(path for path in (directory/"images").iterdir() if path.suffix.lower() in {".jpg",".jpeg",".png"}) if (directory/"images").is_dir() else []
        if not frames:
            failures.extend({
                "status":"missing_official_frames","split":args.split,"video_id":video_id,
                "object_id":str(question["obj_id"]),"expression_id":expression_id,
                "expression":question["question"],"expected_directory":str(directory/"images"),
            } for expression_id,question in sequential)
            continue
        if any(path.suffix.lower()!=".jpg" for path in frames):
            raise RuntimeError(f"SAM3.1 adapter currently requires official JPG frames: {directory}")
        frame_names = [path.stem for path in frames]
        positions = np.linspace(0, len(frames) - 1, num=min(args.evaluation_frames, len(frames)), dtype=int).tolist()
        for expression_id,question in sorted(sequential,key=lambda value:int(value[0])):
            official_object_id=str(question["obj_id"]);object_ids=[int(value.strip()) for value in official_object_id.split(",")]
            action_start,action_end=official_action_window(video_id,question)
            item_object_id=f"{official_object_id}__exp{expression_id}"
            mask_paths=[];present=[];stem_fallback_count=0
            for position in positions:
                canonical=directory/"masks"/f"frame_{position:06d}.png"
                same_stem=directory/"masks"/f"{frame_names[position]}.png"
                source=canonical if canonical.is_file() else same_stem
                if not canonical.is_file() and same_stem.is_file():stem_fallback_count+=1
                target=binary_root/video_id/item_object_id.replace(",","_")/f"frame_{position:06d}.png"
                available=action_start<=position<=action_end and source.is_file()
                if available:
                    mask=np.asarray(Image.open(source).convert("P"));binary=np.isin(mask,object_ids).astype(np.uint8)*255
                    target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(binary).save(target)
                present.append(available);mask_paths.append(str(target))
            expression={
                "expression_id":expression_id,"type":"sequential","text":question["question"],"answer":question["answer"],
                "official_q_type":question["q_type"],"action_start":question["action_start"],"action_end":question["action_end"],
                "official_action_start_index":action_start,"official_action_end_index":action_end,
                "mask_same_stem_fallback_count":stem_fallback_count,
            }
            objects.append({
                "dataset":"groundmore","split":args.split,"video_id":video_id,"object_id":item_object_id,
                "official_object_id":official_object_id,"frame_count":len(frames),"frame_names":frame_names,
                "evaluation_frame_indices":positions,"evaluation_frame_names":[frame_names[index] for index in positions],
                "evaluation_mask_paths":mask_paths,"evaluation_mask_present":present,"expressions":[expression],
            })
    manifest = {
        "dataset": {
            "name": "GroundMoRe", "version": "v2", "split": args.split,
            "source": "https://huggingface.co/datasets/groundmore/GroundMoRe",
            "official_metadata": str(metadata_path), "official_metadata_sha256": digest(metadata_path),
            "official_code_commit": "d5074ab920a86a7ea92ecc69902109f6887f3e10",
            "image_root": str(image_root), "annotation_root": str(data_root / "annotations"),
            "binary_masks_are_evaluation_only": True,
        },
        "selection": {
            "q_type": "official Sequential only", "manual_or_llm_classification": False,
            "evaluation_frames": "20 uniformly sampled official frames (all when fewer)",
            "mask_protocol": "official expression-specific 6fps action window; prefer frame_{index:06d}.png, then pair the archive's actual image/mask stem for v2 naming compatibility; outside/missing masks are zero",
            "gt_read_during_candidate_generation": False,
            "official_sequential_expressions":official_sequential_expressions,
            "available_sequential_expressions":len(objects),
            "missing_official_data_expressions":len(failures),
            "evaluation_gt_visible_expressions":sum(any(item["evaluation_mask_present"]) for item in objects),
            "evaluation_gt_all_zero_expressions":sum(not any(item["evaluation_mask_present"]) for item in objects),
            "same_stem_fallback_expressions":sum(item["expressions"][0]["mask_same_stem_fallback_count"]>0 for item in objects),
            "same_stem_fallback_frames":sum(item["expressions"][0]["mask_same_stem_fallback_count"] for item in objects),
        },
        "objects":objects,"failures":failures,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "objects": len(objects), "expressions": sum(len(item["expressions"]) for item in objects),
        "videos": len({item["video_id"] for item in objects}),"failures":len(failures),"output":str(output),
    }))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--binary-mask-root", required=True); parser.add_argument("--split", choices=("trainval", "test"), required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--evaluation-frames", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

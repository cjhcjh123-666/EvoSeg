"""Cache frozen official SAM3.1 propagation feature maps at the four TDSP anchors."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from projects.evoseg.temporal_compiler.sam31_candidate_protocol import audit_loaded_checkpoint, sha256
from projects.evoseg.temporal_dense_prompt.protocol import stage_endpoints


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def image_tensor(path: Path, device: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((1008, 1008)), dtype=np.float32) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1).half()
    value = (value - 0.5) / 0.5
    return value.unsqueeze(0).to(f"cuda:{device}")


def run(args) -> int:
    sam_repo = Path(args.sam3_repo).resolve(); sys.path.insert(0, str(sam_repo))
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    source = Path(inspect.getfile(build_sam3_multiplex_video_predictor)).resolve()
    if sam_repo not in source.parents: raise RuntimeError(f"non-official SAM import: {source}")
    checkpoint = Path(args.checkpoint).resolve()
    if sha256(checkpoint) != args.expected_checkpoint_sha256: raise RuntimeError("checkpoint SHA mismatch")
    manifest = json.loads(Path(args.manifest).read_text())
    objects = manifest["objects"]
    videos = {}
    for item in objects:
        videos.setdefault(item["video_id"], item)
    videos = [value for index, value in enumerate(sorted(videos.values(), key=lambda row: row["video_id"])) if index % args.num_shards == args.shard_index]
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint), max_num_objects=16, multiplex_count=16, use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
    predictor.model.eval(); audit = audit_loaded_checkpoint(predictor.model, checkpoint)
    started = time.monotonic(); completed = 0; peak = 0
    with torch.inference_mode():
        for item in videos:
            endpoints = stage_endpoints(item["frame_count"])
            for stage_order, endpoint in enumerate(endpoints):
                target = output / f"{item['video_id']}__{endpoint}.npz"
                if target.is_file(): continue
                path = Path(manifest["dataset"]["image_root"]) / item["video_id"] / f"{item['frame_names'][endpoint]}.jpg"
                tensor = image_tensor(path, args.device)
                torch.cuda.reset_peak_memory_stats(args.device); before = time.monotonic()
                # The official 3.1 predictor wraps tracker and detector.  The
                # shared TriHead image encoder lives on the public detector;
                # consume its propagation map without touching prompt internals.
                backbone = predictor.model.detector.backbone.forward_image(tensor)
                feature = backbone["sam2_backbone_out"]["vision_features"][0].detach().float().cpu().numpy().astype(np.float16)
                np.savez_compressed(target, feature=feature, frame_index=np.int32(endpoint), stage_order=np.int32(stage_order))
                torch.cuda.synchronize(args.device); peak = max(peak, torch.cuda.max_memory_allocated(args.device)); completed += 1
            atomic(output / f"STATUS.shard{args.shard_index}.json", {"state":"running","pid":__import__('os').getpid(),"videos_completed":completed // 4,"videos_planned":len(videos),"feature_shape":list(feature.shape),"peak_memory_bytes":peak,"elapsed_seconds":time.monotonic()-started})
    atomic(output / f"STATUS.shard{args.shard_index}.json", {"state":"complete","pid":__import__('os').getpid(),"videos_completed":len(videos),"videos_planned":len(videos),"peak_memory_bytes":peak,"elapsed_seconds":time.monotonic()-started,"ground_truth_read":False,"official_builder_source":str(source),"weight_audit":audit})
    return 0


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',required=True); parser.add_argument('--output',required=True); parser.add_argument('--sam3-repo',required=True); parser.add_argument('--checkpoint',required=True); parser.add_argument('--expected-checkpoint-sha256',required=True); parser.add_argument('--device',type=int,required=True); parser.add_argument('--shard-index',type=int,default=0); parser.add_argument('--num-shards',type=int,default=1); return parser.parse_args()


if __name__ == '__main__': raise SystemExit(run(parse_args()))

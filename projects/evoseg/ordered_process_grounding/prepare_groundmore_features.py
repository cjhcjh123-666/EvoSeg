"""Prepare frozen GroundMoRe Sequential query/candidate T=8 features."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.ordered_process_grounding.prepare_features import (
    batch_region_pool,
    frozen_embedding,
)
from projects.evoseg.temporal_compiler.temporal_matcher_prototype import (
    load_siglip_vision_model,
)


def candidate_index(root: Path, method: str) -> dict[str, dict]:
    result = {}
    for shard in sorted(root.glob("shard-*")):
        path = shard / "candidate_records.jsonl"
        if not path.is_file(): continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "success" or row["prompt_method"] != method: continue
                identity = "/".join((row["dataset"], row["video_id"], str(row["object_id"]), str(row["expression_id"])))
                row["tracks_path"] = str(shard / row["candidate_tracks_path"])
                result[identity] = row
    return result


def run(args) -> None:
    import torch
    from PIL import Image
    from transformers import AutoTokenizer

    manifest = json.loads(Path(args.manifest).read_text())
    objects = {(item["video_id"], str(item["object_id"])): item for item in manifest["objects"]}
    expressions = {
        "/".join((item["dataset"], item["video_id"], str(item["object_id"]), str(exp["expression_id"]))): (item, exp)
        for item in manifest["objects"] for exp in item["expressions"]
    }
    candidates = candidate_index(Path(args.candidate_root), args.prompt_method)
    selected_ids = sorted(expressions)
    videos=sorted({expressions[identity][0]["video_id"] for identity in selected_ids})
    selected_videos={video for index,video in enumerate(videos) if index % args.num_shards==args.shard_index}
    selected_ids=[identity for identity in selected_ids if expressions[identity][0]["video_id"] in selected_videos]
    tokenizer = AutoTokenizer.from_pretrained(args.sa2va_model, trust_remote_code=True)
    embeddings, embedding_audit = frozen_embedding(Path(args.sa2va_model))
    torch.cuda.set_device(args.device)
    vision, processor, config, vision_audit = load_siglip_vision_model(Path(args.vision_checkpoint), f"cuda:{args.device}")
    grid = config.image_size // config.patch_size
    image_root = Path(manifest["dataset"]["image_root"])
    grouped = defaultdict(list)
    for identity in selected_ids: grouped[expressions[identity][0]["video_id"]].append(identity)
    payload=[]; records=[]
    with torch.inference_mode():
        for number,(video_id,keys) in enumerate(sorted(grouped.items()),1):
            item0=expressions[keys[0]][0]; required=set(); positions={}; tracks_by_id={}
            for identity in keys:
                if identity in candidates:
                    tracks=json.loads(Path(candidates[identity]["tracks_path"]).read_text()); tracks_by_id[identity]=tracks
                    p=np.linspace(0,len(tracks["evaluation_frame_indices"])-1,num=args.track_steps,dtype=int).tolist()
                    positions[identity]=p; required.update(tracks["evaluation_frame_indices"][x] for x in p)
                else: positions[identity]=[]
            frame_features={}
            for offset in range(0,len(required),args.image_batch_size):
                indices=sorted(required)[offset:offset+args.image_batch_size]; images=[]
                for index in indices:
                    with Image.open(image_root/video_id/"images"/f"{item0['frame_names'][index]}.jpg") as image:
                        images.append(image.convert("RGB"))
                pixels=processor(images=images,return_tensors="pt")["pixel_values"].to(f"cuda:{args.device}",dtype=next(vision.parameters()).dtype)
                hidden=vision(pixels,output_hidden_states=True).hidden_states[int(vision_audit["selected_hidden_layer"])].float().cpu()
                frame_features.update(dict(zip(indices,hidden)))
            for identity in keys:
                item,expression=expressions[identity]
                token_ids=tokenizer(expression["text"],add_special_tokens=False)["input_ids"][:args.max_query_tokens]
                if not token_ids: token_ids=[tokenizer.unk_token_id]
                query_tokens=embeddings[torch.tensor(token_ids)].to(torch.float16).clone()
                features=[];track_ids=[];selected=[]
                if identity in tracks_by_id:
                    tracks=tracks_by_id[identity]; p=positions[identity];selected=[tracks["evaluation_frame_indices"][x] for x in p]
                    features=batch_region_pool(
                        tracks["tracks"],p,selected,frame_features,(grid,grid)
                    )
                    track_ids=[int(track["track_id"]) for track in tracks["tracks"]]
                array=features.astype(np.float16) if len(features) else np.zeros((0,args.track_steps,config.hidden_size),np.float16)
                payload.append({"identity":identity,"query_tokens":query_tokens,"tracks":torch.from_numpy(array)})
                records.append({
                    "identity":identity,"dataset":item["dataset"],"split":item["split"],"video_id":item["video_id"],
                    "object_id":item["object_id"],"expression_id":expression["expression_id"],"description_type":"sequential",
                    "expression":expression["text"],"candidate_track_ids":track_ids,"selected_frame_indices_t8":selected,
                    "candidate_generation_failure":identity not in candidates,"query_token_count":len(token_ids),
                })
            print(f"videos {number}/{len(grouped)} records={len(records)}",flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output)
    output.with_suffix(".json").write_text(json.dumps({
        "command":[sys.executable,*sys.argv],"manifest":str(Path(args.manifest).resolve()),"candidate_root":str(Path(args.candidate_root).resolve()),
        "prompt_method":args.prompt_method,"ground_truth_read":False,"track_steps":args.track_steps,
        "embedding_audit":embedding_audit,"vision_audit":vision_audit,"records":records,
    },indent=2,ensure_ascii=False)+"\n")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True);p.add_argument("--candidate-root",required=True)
    p.add_argument("--prompt-method",default="raw_expression");p.add_argument("--sa2va-model",required=True);p.add_argument("--vision-checkpoint",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",type=int,required=True);p.add_argument("--track-steps",type=int,default=8)
    p.add_argument("--max-query-tokens",type=int,default=64);p.add_argument("--image-batch-size",type=int,default=8);p.add_argument("--shard-index",type=int,default=0);p.add_argument("--num-shards",type=int,default=1)
    return p.parse_args()


if __name__=="__main__": run(parse_args())

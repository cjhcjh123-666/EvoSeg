"""Convert Ryvos manifest (sampled 5-frame RVOS) to Sa2VA03RefVOS ('default') format.

meta_expressions.json: {"videos": {vid: {"expressions": {eid: {exp, anno_id, obj_id}},
                                        "frames": [frame_id_str, ...]}}}
mask_dict.json: {anno_id: [RLE|None] * n_frames}  (indexed by frame position)
"""
import json, os, re, sys
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

MANIFEST = "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_ryvos/train.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/ryvos"

os.makedirs(OUT, exist_ok=True)
videos = {}
mask_dict = {}
anno_count = 0
skipped_empty = 0
seen_videos = {}

with open(MANIFEST) as f:
    for line in f:
        r = json.loads(line)
        query = (r.get("query") or "").strip()
        if not query:
            skipped_empty += 1
            continue
        vid = r["media_id"].split(".", 1)[1]
        mask_paths = r["mask_paths"]
        if not mask_paths or any(mp is None for mp in mask_paths):
            continue  # skip no-object / negative samples
        # frames from mask filenames: <vid>_<obj>_<frame>.png
        frames = []
        rles = []
        for mp in mask_paths:
            base = os.path.basename(mp)  # e.g. 9fcb310255_1_00000.png
            m = re.match(r".*_(\d+)_(\d+)\.png$", base)
            if not m:
                continue
            obj_id = int(m.group(1))
            frame = m.group(2)
            if not os.path.isfile(mp):
                rles.append(None)
            else:
                a = np.asarray(Image.open(mp).convert("L"))
                a = (a > 127).astype(np.uint8)
                if a.sum() == 0:
                    rles.append(None)
                else:
                    rle = mask_utils.encode(np.asfortranarray(a))
                    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
                    rles.append(rle)
            frames.append(frame)
        if not frames:
            continue
        if vid not in videos:
            videos[vid] = {"frames": frames, "expressions": {}}
        eid = str(len(videos[vid]["expressions"]))
        videos[vid]["expressions"][eid] = {
            "exp": query, "anno_id": [anno_count], "obj_id": [obj_id],
        }
        mask_dict[str(anno_count)] = rles
        anno_count += 1

meta = {"videos": videos}
with open(os.path.join(OUT, "meta_expressions.json"), "w") as f:
    json.dump(meta, f)
with open(os.path.join(OUT, "mask_dict.json"), "w") as f:
    json.dump(mask_dict, f)
print(f"ryvos: {len(videos)} videos, {anno_count} expressions (skipped {skipped_empty} empty) -> {OUT}")

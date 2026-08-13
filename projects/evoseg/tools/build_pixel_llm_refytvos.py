"""Build Ref-YT-VOS mask_dict.pkl for Sa2VA03RefVOS ('refytvos' type).

Annotations are indexed masks (pixel value = instance id). meta obj_id is a
1-based per-video object index -> map by first-appearance order of instances.
mask_dict = {anno_count_str: [RLE|None] * n_frames}, indexed by frame position.
"""
import json, os, pickle, sys
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

META = "/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/meta_expressions/train/meta_expressions.json"
ANN_ROOT = "/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/train/Annotations"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/ref_youtube_vos/mask_dict.pkl"

def video_instance_order(vid_dir, frames):
    """Instance ids ordered by first appearance across frames."""
    seen = []
    for frame in frames:
        png = os.path.join(vid_dir, frame + ".png")
        if not os.path.isfile(png):
            continue
        a = np.asarray(Image.open(png).convert("L"))
        vals = np.unique(a)
        for v in vals:
            if v != 0 and v not in seen:
                seen.append(int(v))
    return seen

meta = json.load(open(META))["videos"]
mask_dict = {}
anno_count = 0
empty_all = 0
for vid_name in meta:
    vid = meta[vid_name]
    frames = sorted(vid["frames"])
    vid_dir = os.path.join(ANN_ROOT, vid_name)
    order = video_instance_order(vid_dir, frames)  # obj index 1 -> order[0]
    for exp_id in sorted(vid["expressions"].keys()):
        obj_idx = int(vid["expressions"][exp_id]["obj_id"])
        inst = order[obj_idx - 1] if 0 < obj_idx <= len(order) else None
        per_frame = []
        for frame in frames:
            png = os.path.join(vid_dir, frame + ".png")
            if not os.path.isfile(png) or inst is None:
                per_frame.append(None)
                continue
            a = np.asarray(Image.open(png).convert("L"))
            m = (a == inst).astype(np.uint8)
            if m.sum() == 0:
                per_frame.append(None)
            else:
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
                per_frame.append(rle)
        if all(x is None for x in per_frame):
            empty_all += 1
        mask_dict[str(anno_count)] = per_frame
        anno_count += 1
        if anno_count % 1000 == 0:
            print(f"  ...{anno_count} annos processed", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(mask_dict, f)
print(f"refytvos mask_dict: {anno_count} annos -> {OUT} (fully-empty: {empty_all})")

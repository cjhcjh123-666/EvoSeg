"""Convert ReasonSeg train (HF parquet) to Pixel-LLM finetune format.

Output: {out}/annotations.json [{image, mask:[[flat_poly]], text:[query]}] + {out}/images/*.jpg
"""
import json, os, sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

SRC_DIR = "/9950backfile/chenjiahui/evo_artifacts/datasets/reasonseg_train"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/reason_seg"

def mask_to_polygons(mask: np.ndarray, max_pts=256):
    a = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        if len(c) > max_pts:
            idx = np.linspace(0, len(c)-1, max_pts).astype(int)
            c = c[idx]
        polys.append(c.flatten().tolist())
    return polys

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
anns = []
n = 0
for fn in sorted(os.listdir(SRC_DIR)):
    if not fn.endswith(".parquet"):
        continue
    df = pd.read_parquet(os.path.join(SRC_DIR, fn))
    for _, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        text = row["text"]
        mask = row["mask"]
        mask = np.asarray(mask)
        # HF object-array: rows of bool arrays -> reconstruct full (H, W) mask
        if mask.dtype == object:
            mask = np.stack([np.asarray(m) for m in mask])
        if mask.ndim == 3:
            mask = mask[0]
        img_name = f"reasonseg_{row['image_id']}.jpg"
        with open(os.path.join(OUT, "images", img_name), "wb") as f:
            f.write(img_bytes)
        polys = mask_to_polygons(mask)
        if not polys:
            continue
        anns.append({"image": img_name, "mask": [polys], "text": [text]})
        n += 1

with open(os.path.join(OUT, "annotations.json"), "w") as f:
    json.dump(anns, f)
print(f"reasonseg: {n} samples -> {OUT}")

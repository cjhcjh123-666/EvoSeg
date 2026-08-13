"""Build a tiny finetune-format smoke dataset from the EvoVILA RefCOCO manifests.

Output: {out_dir}/annotations.json + {out_dir}/images/*.jpg
Format matches Sa2VAFinetuneDataset: [{image, mask: [[x,y,...]], text: [query]}]
"""
import json, os, shutil, sys
import numpy as np
from PIL import Image
import cv2

MANIFEST = "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_refcoco/train.jsonl"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/9950backfile/chenjiahui/evoseg_smoke_data"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16

def mask_to_polygons(png_path, max_pts=256):
    a = np.array(Image.open(png_path).convert("L"))
    a = (a > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        # downsample if too long
        if len(c) > max_pts:
            idx = np.linspace(0, len(c)-1, max_pts).astype(int)
            c = c[idx]
        # EvoSeg: flat polygon [x1,y1,x2,y2,...] (frPyObjects accepts [flat_poly])
        poly = c.flatten().tolist()
        polys.append(poly)
    return polys

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
anns = []
count = 0
with open(MANIFEST) as f:
    for line in f:
        if count >= N:
            break
        item = json.loads(line)
        if item.get("control_kind") != "positive":
            continue
        if not item.get("target_presence", [True])[0]:
            continue
        img_src = item["media_path"]
        mask_path = item["mask_paths"][0]
        query = item["query"]
        img_name = os.path.basename(img_src)
        dst_img = os.path.join(OUT, "images", img_name)
        if not os.path.exists(dst_img):
            shutil.copy(img_src, dst_img)
        polys = mask_to_polygons(mask_path)
        if not polys:
            continue
        # mask = list of objects; each object = list of flat polygons
        anns.append({"image": img_name, "mask": [polys], "text": [query]})
        count += 1

with open(os.path.join(OUT, "annotations.json"), "w") as f:
    json.dump(anns, f)
print(f"built {len(anns)} smoke samples -> {OUT}")

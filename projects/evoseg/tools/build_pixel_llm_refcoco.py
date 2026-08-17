"""Convert local HF-parquet RefCOCO/+/g into Pixel-LLM (Sa2VA) data format.

Output layout (DATA_ROOT):
  ref_seg/refcoco/{instances.json, refs(unc).p, coco2014/train2014 -> images}
  ref_seg/refcoco+/...
  ref_seg/refcocog/{instances.json, refs(umd).p, ...}

instances.json: COCO {"images":[...], "annotations":[...]} (referred objects only)
refs pkl      : refer-format list of {ref_id, ann_id, image_id, category_id,
                split, sentences:[{tokens,raw,sent_id}], sent_ids}
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd

PARQUET_ROOT = "/9950backfile/chenjiahui/evo_artifacts/datasets/refcoco_hf"
IMG_ROOT = "/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014"
DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data"

VARIANTS = [  # (variant, out_dir, refs_filename) -- Pixel-LLM uses 'refcoco+'
    ("refcoco", "refcoco", "refs(unc).p"),
    ("refcocoplus", "refcoco+", "refs(unc).p"),
    ("refcocog", "refcocog", "refs(umd).p"),
]

def build(variant, out_dir, refs_name):
    parq_files = sorted(f for f in os.listdir(os.path.join(PARQUET_ROOT, variant))
                        if f.endswith(".parquet"))
    out = os.path.join(DATA_ROOT, "ref_seg", out_dir)
    os.makedirs(out, exist_ok=True)

    images, annotations, refs = {}, {}, []
    for parq in parq_files:
        df = pd.read_parquet(os.path.join(PARQUET_ROOT, variant, parq))
        for _, row in df.iterrows():
            ann = json.loads(row["raw_anns"]) if isinstance(row["raw_anns"], str) else row["raw_anns"]
            img = json.loads(row["raw_image_info"]) if isinstance(row["raw_image_info"], str) else row["raw_image_info"]
            sentences = []
            for s in row["sentences"]:
                sentences.append({
                    "tokens": [str(t) for t in s["tokens"]],
                    "raw": s["raw"],
                    "sent": s["sent"],
                    "sent_id": int(s["sent_id"]),
                })
            ref = {
                "ref_id": int(row["ref_id"]),
                "ann_id": int(row["ann_id"]),
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "split": str(row["split"]),
                "sentences": sentences,
                "sent_ids": [int(x) for x in row["sent_ids"]],
            }
            refs.append(ref)
            ann_id = int(ann["id"])
            annotations[ann_id] = {
                "id": ann_id,
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "segmentation": ann["segmentation"],
                "area": float(ann["area"]),
                "bbox": [float(x) for x in ann["bbox"]],
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            images[int(img["id"])] = {
                "id": int(img["id"]),
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
            }

    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
        {"id": 5, "name": "airplane", "supercategory": "vehicle"},
        {"id": 6, "name": "bus", "supercategory": "vehicle"},
        {"id": 7, "name": "train", "supercategory": "vehicle"},
        {"id": 8, "name": "truck", "supercategory": "vehicle"},
        {"id": 9, "name": "boat", "supercategory": "vehicle"},
        {"id": 10, "name": "traffic light", "supercategory": "outdoor"},
        {"id": 11, "name": "fire hydrant", "supercategory": "outdoor"},
        {"id": 13, "name": "stop sign", "supercategory": "outdoor"},
        {"id": 14, "name": "parking meter", "supercategory": "outdoor"},
        {"id": 15, "name": "bench", "supercategory": "outdoor"},
        {"id": 16, "name": "bird", "supercategory": "animal"},
        {"id": 17, "name": "cat", "supercategory": "animal"},
        {"id": 18, "name": "dog", "supercategory": "animal"},
        {"id": 19, "name": "horse", "supercategory": "animal"},
        {"id": 20, "name": "sheep", "supercategory": "animal"},
        {"id": 21, "name": "cow", "supercategory": "animal"},
        {"id": 22, "name": "elephant", "supercategory": "animal"},
        {"id": 23, "name": "bear", "supercategory": "animal"},
        {"id": 24, "name": "zebra", "supercategory": "animal"},
        {"id": 25, "name": "giraffe", "supercategory": "animal"},
        {"id": 27, "name": "backpack", "supercategory": "accessory"},
        {"id": 28, "name": "umbrella", "supercategory": "accessory"},
        {"id": 31, "name": "handbag", "supercategory": "accessory"},
        {"id": 32, "name": "tie", "supercategory": "accessory"},
        {"id": 33, "name": "suitcase", "supercategory": "accessory"},
        {"id": 34, "name": "frisbee", "supercategory": "sports"},
        {"id": 35, "name": "skis", "supercategory": "sports"},
        {"id": 36, "name": "snowboard", "supercategory": "sports"},
        {"id": 37, "name": "sports ball", "supercategory": "sports"},
        {"id": 38, "name": "kite", "supercategory": "sports"},
        {"id": 39, "name": "baseball bat", "supercategory": "sports"},
        {"id": 40, "name": "baseball glove", "supercategory": "sports"},
        {"id": 41, "name": "skateboard", "supercategory": "sports"},
        {"id": 42, "name": "surfboard", "supercategory": "sports"},
        {"id": 43, "name": "tennis racket", "supercategory": "sports"},
        {"id": 44, "name": "bottle", "supercategory": "kitchen"},
        {"id": 46, "name": "wine glass", "supercategory": "kitchen"},
        {"id": 47, "name": "cup", "supercategory": "kitchen"},
        {"id": 48, "name": "fork", "supercategory": "kitchen"},
        {"id": 49, "name": "knife", "supercategory": "kitchen"},
        {"id": 50, "name": "spoon", "supercategory": "kitchen"},
        {"id": 51, "name": "bowl", "supercategory": "kitchen"},
        {"id": 52, "name": "banana", "supercategory": "food"},
        {"id": 53, "name": "apple", "supercategory": "food"},
        {"id": 54, "name": "sandwich", "supercategory": "food"},
        {"id": 55, "name": "orange", "supercategory": "food"},
        {"id": 56, "name": "broccoli", "supercategory": "food"},
        {"id": 57, "name": "carrot", "supercategory": "food"},
        {"id": 58, "name": "hot dog", "supercategory": "food"},
        {"id": 59, "name": "pizza", "supercategory": "food"},
        {"id": 60, "name": "donut", "supercategory": "food"},
        {"id": 61, "name": "cake", "supercategory": "food"},
        {"id": 62, "name": "chair", "supercategory": "furniture"},
        {"id": 63, "name": "couch", "supercategory": "furniture"},
        {"id": 64, "name": "potted plant", "supercategory": "furniture"},
        {"id": 65, "name": "bed", "supercategory": "furniture"},
        {"id": 67, "name": "dining table", "supercategory": "furniture"},
        {"id": 70, "name": "toilet", "supercategory": "furniture"},
        {"id": 72, "name": "tv", "supercategory": "electronic"},
        {"id": 73, "name": "laptop", "supercategory": "electronic"},
        {"id": 74, "name": "mouse", "supercategory": "electronic"},
        {"id": 75, "name": "remote", "supercategory": "electronic"},
        {"id": 76, "name": "keyboard", "supercategory": "electronic"},
        {"id": 77, "name": "cell phone", "supercategory": "electronic"},
        {"id": 78, "name": "microwave", "supercategory": "appliance"},
        {"id": 79, "name": "oven", "supercategory": "appliance"},
        {"id": 80, "name": "toaster", "supercategory": "appliance"},
        {"id": 81, "name": "sink", "supercategory": "appliance"},
        {"id": 82, "name": "refrigerator", "supercategory": "appliance"},
        {"id": 84, "name": "book", "supercategory": "indoor"},
        {"id": 85, "name": "clock", "supercategory": "indoor"},
        {"id": 86, "name": "vase", "supercategory": "indoor"},
        {"id": 87, "name": "scissors", "supercategory": "indoor"},
        {"id": 88, "name": "teddy bear", "supercategory": "indoor"},
        {"id": 89, "name": "hair drier", "supercategory": "indoor"},
        {"id": 90, "name": "toothbrush", "supercategory": "indoor"},
    ]
    instances = {"images": list(images.values()), "annotations": list(annotations.values()),
                 "categories": categories}
    with open(os.path.join(out, "instances.json"), "w") as f:
        json.dump(instances, f)
    with open(os.path.join(out, refs_name), "wb") as f:
        pickle.dump(refs, f)

    link = os.path.join(out, "coco2014", "train2014")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.islink(link):
        os.symlink(IMG_ROOT, link)
    splits = {}
    for r in refs:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"[{variant}] refs={len(refs)} splits={splits} -> {out}", flush=True)

    missing = sum(1 for a in annotations.values()
                  if not os.path.isfile(os.path.join(IMG_ROOT, images[a["image_id"]]["file_name"])))
    print(f"[{variant}] missing images: {missing}", flush=True)

for v, od, rn in VARIANTS:
    build(v, od, rn)
print("ALL DONE")

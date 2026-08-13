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
    parq = os.path.join(PARQUET_ROOT, variant, "train.parquet")
    df = pd.read_parquet(parq)
    out = os.path.join(DATA_ROOT, "ref_seg", out_dir)
    os.makedirs(out, exist_ok=True)

    images, annotations, refs = {}, {}, []
    for _, row in df.iterrows():
        ann = json.loads(row["raw_anns"]) if isinstance(row["raw_anns"], str) else row["raw_anns"]
        img = json.loads(row["raw_image_info"]) if isinstance(row["raw_image_info"], str) else row["raw_image_info"]
        sentences = []
        for s in row["sentences"]:
            sentences.append({
                "tokens": [str(t) for t in s["tokens"]],
                "raw": s["raw"],
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
            "segmentation": ann["segmentation"],  # list of flat polygons
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

    # COCO instances.json
    instances = {"images": list(images.values()), "annotations": list(annotations.values())}
    with open(os.path.join(out, "instances.json"), "w") as f:
        json.dump(instances, f)
    # refs pkl
    with open(os.path.join(out, refs_name), "wb") as f:
        pickle.dump(refs, f)

    # image dir symlink
    link = os.path.join(out, "coco2014", "train2014")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.islink(link):
        os.symlink(IMG_ROOT, link)
    print(f"[{variant}] refs={len(refs)} anns={len(annotations)} imgs={len(images)} "
          f"-> {out} (refs: {refs_name})")

    # sanity: every referenced file exists
    missing = 0
    for a in annotations.values():
        fn = images[a["image_id"]]["file_name"]
        if not os.path.isfile(os.path.join(IMG_ROOT, fn)):
            missing += 1
    print(f"[{variant}] missing images: {missing}")

for v, od, rn in VARIANTS:
    build(v, od, rn)
print("ALL DONE")

"""Convert gRefCOCO (GRES, target-present) to Pixel-LLM finetune format.

- single & multi-target refs -> one entry with combined polygons
- no-target refs are SKIPPED here (handled by the faithful GRES dataset later)
Output: {out}/annotations.json [{image, mask:[[flat_poly,...]], text:[query]}] + {out}/images symlink
"""
import json, os, sys

SRC = "/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco"
IMG_ROOT = "/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/grefcoco"

instances = json.load(open(os.path.join(SRC, "instances.json")))
anns = {a["id"]: a for a in instances["annotations"]}
images = {im["id"]: im for im in instances["images"]}
refs = json.load(open(os.path.join(SRC, "grefs_unc.json")))

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
link = os.path.join(OUT, "images", "train2014")
if not os.path.islink(link):
    os.symlink(IMG_ROOT, link)

entries = []
n_target = n_multi = n_no = 0
seen = set()
for r in refs:
    if r["no_target"]:
        n_no += 1
        continue
    img = images[r["image_id"]]
    file_name = img["file_name"]
    if file_name in seen:
        continue
    seen.add(file_name)
    polys = []
    for aid in r["ann_id"]:
        seg = anns[aid]["segmentation"]
        if seg:
            polys.extend(seg)
    if not polys:
        continue
    query = r["sentences"][0]["raw"]
    if query.endswith("."):
        query = query[:-1]
    if len(r["ann_id"]) > 1:
        n_multi += 1
    else:
        n_target += 1
    entries.append({"image": file_name, "mask": [polys], "text": [query]})

with open(os.path.join(OUT, "annotations.json"), "w") as f:
    json.dump(entries, f)
print(f"grefcoco target-present: {len(entries)} entries "
      f"(single={n_target}, multi={n_multi}, skipped_no_target={n_no}) -> {OUT}")

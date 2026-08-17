"""MeViS J&F at NATIVE resolution (official-style), 4 procs to bound memory."""
import json, sys, numpy as np
from multiprocessing import Pool
from pycocotools import mask as mask_utils
from scipy import ndimage

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/9950backfile/chenjiahui/EvoSeg/work_dirs/foobar/MEVIS_U/results.json"
META = "/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/valid_u/meta_expressions_v2.json"
MASK = "/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/valid_u/mask_dict.json"

def db_eval_boundary(fg, gt, bound_th=0.008):
    fg = fg.astype(np.float64); gt = gt.astype(np.float64)
    if np.sum(fg) == 0 and np.sum(gt) == 0: return 1.0
    if np.sum(fg) == 0 or np.sum(gt) == 0: return 0.0
    fg_b = (fg - ndimage.binary_erosion(fg > 0).astype(float)) > 0
    gt_b = (gt - ndimage.binary_erosion(gt > 0).astype(float)) > 0
    fg_d = ndimage.distance_transform_edt(np.logical_not(fg_b))
    gt_d = ndimage.distance_transform_edt(np.logical_not(gt_b))
    max_dist = bound_th * np.sqrt(fg.shape[0]**2 + fg.shape[1]**2)
    fg_d = np.minimum(fg_d, max_dist); gt_d = np.minimum(gt_d, max_dist)
    prec = np.mean(fg_d[gt_b > 0]) if np.sum(gt_b) > 0 else 0.0
    rec  = np.mean(gt_d[fg_b > 0]) if np.sum(fg_b) > 0 else 0.0
    prec = 1.0 - prec / max_dist; rec = 1.0 - rec / max_dist
    return 2 * prec * rec / (prec + rec + 1e-10)

def decode_bool(rle):
    if rle is None: return None
    return mask_utils.decode(rle).astype(bool)

def eval_video(args):
    vid, exps, meta, mask_dict = args
    out = []
    for exp_id, item in exps.items():
        exp_info = meta.get(exp_id)
        if exp_info is None: continue
        anno_ids = [str(a) for a in exp_info["anno_id"]]
        pred = item["prediction_masks"]
        n = len(pred)
        Js, Fs = [], []
        for fi in range(n):
            p = decode_bool(pred[fi])
            if p is None: continue
            union = None
            for aid in anno_ids:
                if aid in mask_dict and fi < len(mask_dict[aid]) and mask_dict[aid][fi] is not None:
                    m = decode_bool(mask_dict[aid][fi])
                    if m is None: continue
                    union = m if union is None else (union | m)
            if union is None or union.sum() == 0: continue
            inter = np.logical_and(p, union).sum(); u = np.logical_or(p, union).sum()
            Js.append(inter / (u + 1e-10))
            Fs.append(db_eval_boundary(p, union))
        if Js:
            out.append((np.mean(Js), np.mean(Fs)))
    return out

results = json.load(open(RESULTS))
meta_all = json.load(open(META))["videos"]
mask_dict = json.load(open(MASK))
tasks = [(v, e, meta_all[v]["expressions"], mask_dict) for v, e in results.items() if v in meta_all]
print(f"tasks: {len(tasks)}", flush=True)
pairs = []
with Pool(processes=4) as pool:
    for i, res in enumerate(pool.imap_unordered(eval_video, tasks)):
        pairs.extend(res)
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(tasks)} videos", flush=True)
all_J = [p[0] for p in pairs]; all_F = [p[1] for p in pairs]
print(f"evaluated {len(pairs)} pairs", flush=True)
print(f"Mean J: {np.mean(all_J):.4f}", flush=True)
print(f"Mean F: {np.mean(all_F):.4f}", flush=True)
print(f"J&F:    {(np.mean(all_J)+np.mean(all_F))/2:.4f}", flush=True)

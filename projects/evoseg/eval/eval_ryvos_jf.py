"""Ref-YT-VOS J&F (official-style Jaccard + Contour F), native resolution."""
import json, sys, os, numpy as np
from multiprocessing import Pool
from pycocotools import mask as mask_utils
from scipy import ndimage
from PIL import Image

RESULTS = sys.argv[1] if len(sys.argv) > 1 else None
ANNO = "/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/Annotations"

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
    vid, exps = args
    out = []
    for exp_id, item in exps.items():
        pred = item["prediction_masks"]
        frames = item["frames"]
        inst_dir = os.path.join(ANNO, vid, str(exp_id))
        Js, Fs = [], []
        try:
            for fi, fr in enumerate(frames):
                p = decode_bool(pred[fi]) if fi < len(pred) else None
                if p is None: continue
                gt_path = os.path.join(inst_dir, fr + ".png")
                if not os.path.exists(gt_path): continue
                gt = np.array(Image.open(gt_path).convert('L')) > 0
                if gt.sum() == 0: continue
                if p.shape != gt.shape:
                    # Ref-YT-VOS has videos with inconsistent frame sizes;
                    # resize prediction to GT resolution (official-style).
                    from PIL import Image as _I
                    p = np.array(_I.fromarray(p.astype(np.uint8) * 255).resize(
                        (gt.shape[1], gt.shape[0]), _I.NEAREST)) > 0
                inter = np.logical_and(p, gt).sum(); u = np.logical_or(p, gt).sum()
                Js.append(inter / (u + 1e-10))
                Fs.append(db_eval_boundary(p, gt))
        except Exception as e:
            print(f"  [warn] {vid}/{exp_id}: {e}", flush=True)
            continue
        if Js:
            out.append((np.mean(Js), np.mean(Fs)))
    return out

results = json.load(open(RESULTS))
tasks = list(results.items())
print(f"tasks: {len(tasks)}", flush=True)
pairs = []
with Pool(processes=4) as pool:
    for i, res in enumerate(pool.imap_unordered(eval_video, tasks)):
        pairs.extend(res)
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(tasks)} videos", flush=True)
all_J = [p[0] for p in pairs]; all_F = [p[1] for p in pairs]
print(f"evaluated {len(pairs)} expressions", flush=True)
print(f"Mean J: {np.mean(all_J):.4f}", flush=True)
print(f"Mean F: {np.mean(all_F):.4f}", flush=True)
print(f"J&F:    {(np.mean(all_J)+np.mean(all_F))/2:.4f}", flush=True)

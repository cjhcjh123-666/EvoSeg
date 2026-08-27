"""Third feature pass: per-frame mask geometry (area / centroid / bbox).

For each [SEG] case, propagate with the cached lang and save, per frame:
  area_frac   : mask probability mass fraction of the frame
  cx, cy      : mask centroid (fractional coords)
  bw, bh      : bbox size (fractional)

These capture the 'SAM2 lost the track' signature (mask area grows / centroid
drifts / bbox explodes when the referent leaves) -- orthogonal to appearance.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
MODELS = {
  'teg4b': '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG',
  'mt4b': '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-MultiTask',
  'faithful4b': '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-Faithful',
}
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/train/JPEGImages')
MAN = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
       'faithfulness_train.json')
FEATDIR = '/9950backfile/chenjiahui/evo_artifacts/data/et_features'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir', default=FEATDIR)
    ap.add_argument('--outdir', default=FEATDIR)
    ap.add_argument('--model', default='teg4b', choices=list(MODELS.keys()))
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    os.makedirs(args.outdir, exist_ok=True)

    cases = json.load(open(MAN))['cases']
    by_vid = {}
    for c in cases:
        by_vid.setdefault(c['video_id'], c)

    files = sorted([f for f in os.listdir(args.feat_dir) if f.startswith('et_features_rank')])
    my_files = [f for i, f in enumerate(files) if i % world == rank]

    model = AutoModel.from_pretrained(
        MODELS[args.model], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODELS[args.model], trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODELS[args.model], trust_remote_code=True)

    out_path = os.path.join(args.outdir, f'geom_rank{rank}.pt')
    res = []
    if os.path.exists(out_path):
        res = torch.load(out_path, map_location='cpu')
        print(f'[rank{rank}] resume {len(res)}', flush=True)
    done = len(res)
    t0 = time.time(); n_total = 0
    for fi, f in enumerate(my_files):
        data = torch.load(os.path.join(args.feat_dir, f), map_location='cpu')
        for ri, r in enumerate(data):
            n_total += 1
            if n_total - 1 < done:
                continue
            if not r.get('seg'):
                continue
            vid = r['vid']; query = r['query']
            c = by_vid.get(vid)
            if c is None:
                continue
            frames = [Image.open(os.path.join(JPEGROOT, vid, fr + '.jpg')).convert('RGB')
                      for fr in c['frames']]
            extra_pixel_values = []
            for fr in frames:
                g = model.extra_image_processor.apply_image(np.array(fr))
                extra_pixel_values.append(torch.from_numpy(g).permute(2, 0, 1).contiguous())
            g_pv = torch.stack([model.grounding_encoder.preprocess_image(p)
                                for p in extra_pixel_values]).to(torch.bfloat16)
            lang = r['lang'].float().cuda()
            with torch.no_grad():
                ss = model.grounding_encoder.get_sam2_embeddings(g_pv.cuda())
                nf = min(5, len(frames))
                masks = model.grounding_encoder.language_embd_inference(
                    ss, [lang] * nf)                       # [T,1,H,W] logits
                mp = masks.sigmoid()                        # [T,1,H,W]
                B = mp.shape[0]; H = mp.shape[2]; W = mp.shape[3]
                mpf = mp.reshape(B, H * W)                     # [T, HW]
                mass = mpf.sum(dim=1)                       # [T]
                ys, xs = torch.meshgrid(torch.arange(H).float().cuda(),
                                        torch.arange(W).float().cuda(), indexing='ij')
                xs = xs.reshape(-1); ys = ys.reshape(-1)
                s = mpf.sum(dim=1, keepdim=True).clamp(min=1e-5)
                cx = (mpf * xs.unsqueeze(0)).sum(dim=1) / s.squeeze(1)  # [T]
                cy = (mpf * ys.unsqueeze(0)).sum(dim=1) / s.squeeze(1)
                thr = 0.5
                hard = mpf > thr                            # [T,HW]
                row_any = hard.any(dim=1)                   # [T]
                xs_h = xs.unsqueeze(0).expand(B, -1).clone(); xs_h[~hard] = 1e9
                minx = xs_h.min(dim=1).values; maxx = xs_h.max(dim=1).values
                ys_h = ys.unsqueeze(0).expand(B, -1).clone(); ys_h[~hard] = 1e9
                miny = ys_h.min(dim=1).values; maxy = ys_h.max(dim=1).values
                bw = torch.where(row_any, (maxx - minx) / W, torch.zeros_like(mass))
                bh = torch.where(row_any, (maxy - miny) / H, torch.zeros_like(mass))
                area_frac = mass / (H * W)
            res.append({'vid': vid, 'query': query[:40],
                        'area': area_frac.cpu().float(),
                        'cx': (cx / W).cpu().float(),
                        'cy': (cy / H).cpu().float(),
                        'bw': bw.cpu().float(), 'bh': bh.cpu().float()})
            if n_total % 200 == 0:
                torch.save(res, out_path)
                el = time.time() - t0
                print(f'[rank{rank}] {n_total} {el/n_total:.2f}s/case', flush=True)
        torch.save(res, out_path)
    print(f'[rank{rank}] DONE {len(res)} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()

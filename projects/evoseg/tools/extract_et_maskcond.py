"""Second feature pass: per-frame mask-conditioned SAM2 features.

For every [SEG] case in the cached et_features files, run the real SAM2
propagation (with the cached [SEG] lang) and pool the SAM2 backbone features
under the propagated mask (soft mask weights). This gives mask_cond [T,256]:
when the referent leaves / the mask drifts to a lookalike, the features under
the mask change -> strong e_t signal (unlike the global mean-pooled feat).

Output: <outdir>/maskcond_rank{r}.pt  list of {vid, query[:40], mask_cond [T,256]}
"""
import argparse
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
  'vf4b': '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-VideoFaithful',
  'faithful4b': '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-Faithful',
}
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/train/JPEGImages')
MAN = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
       'faithfulness_train.json')
FEATDIR = '/9950backfile/chenjiahui/evo_artifacts/data/et_features'
CH = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir', default=FEATDIR)
    ap.add_argument('--outdir', default='/9950backfile/chenjiahui/evo_artifacts/data/et_features')
    ap.add_argument('--model', default='teg4b', choices=list(MODELS.keys()))
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    os.makedirs(args.outdir, exist_ok=True)

    cases = json.load(open(MAN))['cases']
    # cache index: vid -> case (first matching) for frame lists
    by_vid = {}
    for c in cases:
        by_vid.setdefault(c['video_id'], c)

    # load cached features (only our share)
    files = sorted([f for f in os.listdir(args.feat_dir) if f.startswith('et_features_rank')])
    my_files = [f for i, f in enumerate(files) if i % world == rank]

    model = AutoModel.from_pretrained(
        MODELS[args.model], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODELS[args.model], trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODELS[args.model], trust_remote_code=True)

    out_path = os.path.join(args.outdir, f'maskcond_rank{rank}.pt')
    res = []
    if os.path.exists(out_path):
        res = torch.load(out_path, map_location='cpu')
        print(f'[rank{rank}] resume {len(res)}', flush=True)
    done = len(res)
    t0 = time.time()
    n_total = 0
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
            T = len(frames)
            extra_pixel_values = []
            for fi2, fr in enumerate(frames):
                g = model.extra_image_processor.apply_image(np.array(fr))
                extra_pixel_values.append(torch.from_numpy(g).permute(2, 0, 1).contiguous())
            g_pv = torch.stack([model.grounding_encoder.preprocess_image(p)
                                for p in extra_pixel_values]).to(torch.bfloat16)
            lang = r['lang'].float().cuda()  # [1,256]
            with torch.no_grad():
                sam_states = model.grounding_encoder.get_sam2_embeddings(g_pv.cuda())
                nf = min(5, T)
                masks = model.grounding_encoder.language_embd_inference(
                    sam_states, [lang] * nf)          # [T,1,Hl,Wl] logits
                feats = model.grounding_encoder.sam2_model.forward_image(g_pv.cuda())
                _, vf, _, _ = model.grounding_encoder.sam2_model._prepare_backbone_features(feats)
                vfeat = vf[-1]                         # [HW,T,C]
                HW, TT, C = vfeat.shape
                H = int(HW ** 0.5)
                vf_2d = vfeat.permute(1, 2, 0).view(TT, C, H, H).float()  # [T,C,H,H]
                # mask prob at feature resolution
                mask_prob = masks.sigmoid()            # [T,1,Hl,Wl]
                mpr = torch.nn.functional.interpolate(
                    mask_prob, size=(H, H), mode='bilinear', align_corners=False)  # [T,1,H,H]
                mpr = mpr.squeeze(1)                   # [T,H,H]
                # soft mask-weighted pool
                num = (vf_2d * mpr.unsqueeze(1)).sum(dim=(-2, -1))   # [T,C]
                den = mpr.sum(dim=(-2, -1)).clamp(min=1e-5)          # [T]
                mask_cond = (num / den.unsqueeze(-1)).cpu()          # [T,C]
            res.append({'vid': vid, 'query': query[:40],
                        'mask_cond': mask_cond.float()})
            if (n_total) % 25 == 0:
                el = time.time() - t0
                print(f'[rank{rank}] {n_total} cases {el/n_total:.2f}s/case '
                      f'eta={(n_total*0 + 1)*0:.0f}', flush=True)
            if n_total % 200 == 0:
                torch.save(res, out_path)
        torch.save(res, out_path)
    print(f'[rank{rank}] DONE {len(res)} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()

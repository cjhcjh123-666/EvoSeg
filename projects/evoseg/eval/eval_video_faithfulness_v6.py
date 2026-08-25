"""Eval v6 head vs v5 head gating on the faithfulness benchmark.

Loads model (B+ v5 head for predict_forward feature extraction), then
recomputes e_t with the v6 head and gates raw masks. Compares:
  v5 (B+ v5 head gate) vs v6 (new head gate).
"""
import argparse, datetime, json, os, time
import torch, torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MANIFEST = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_valid.json'
JPEGROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'

class V6Head(nn.Module):
    def __init__(self, vlm_dim=2560, mask_dim=256, lang_dim=256, hidden=512, n_layers=1):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, 256)
        self.fc_in = nn.Sequential(
            nn.Linear(256 + mask_dim + 256 + 256 + 3 + lang_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, vlm, mask_cond, geom, anchor_mc, anchor_vlm, lang):
        B, T, C = vlm.shape
        v = self.vlm_proj(vlm)
        av = self.vlm_proj(anchor_vlm)
        dv = v - av.expand(B, T, -1)
        dmc = mask_cond - anchor_mc.expand(B, T, -1)
        x = torch.cat([v, mask_cond, dmc, dv, geom, lang.expand(B, T, -1)], dim=-1)
        x = self.fc_in(x)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('--manifest', default=MANIFEST)
    p.add_argument('--v6-ckpt', default='/tmp/v6_head.pt')
    p.add_argument('--save', default=None)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--cat-filter', nargs='*', default=None)
    p.add_argument('--thr', type=float, default=0.5)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=180))
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank)

    manifest = json.load(open(args.manifest))
    cases = manifest['cases']
    if args.cat_filter:
        cases = [c for c in cases if c['category'] in args.cat_filter]
    if args.max_cases:
        cases = cases[:args.max_cases]
    my_cases = [c for i, c in enumerate(cases) if i % world == rank]
    if rank == 0:
        print(f'cases={len(cases)} per_rank={len(my_cases)}', flush=True)

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, use_flash_attn=True,
                                      trust_remote_code=True).eval().cuda()
    model.load_temporal_head()  # v5 head (feature extraction uses it; we overwrite e_t)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    ck = torch.load(args.v6_ckpt, map_location='cpu')
    v6 = V6Head().float().eval().to(f'cuda:{rank}')
    v6.load_state_dict(ck['state_dict'])
    print(f'[v6] loaded {args.v6_ckpt} stats={ck.get("stats")}', flush=True)

    results = []
    t0 = time.time()
    for ci, c in enumerate(my_cases):
        vid = c['video_id']
        frames = [Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
                  for f in c['frames']]
        text = f'<image>\n Please segment {c["query"]} in this video.'
        n = len(frames)
        with torch.no_grad():
            out = model.predict_forward(video=frames, text=text, tokenizer=tok, processor=proc,
                                        vlm_all_frames=True)
        raw = out.get('raw_masks'); mc = out.get('mask_cond'); geom = out.get('geom')
        lg = out.get('lang'); vf = getattr(model, '_vlm_feat', None)
        raw_p = [False] * n
        if raw is not None:
            for t in range(min(n, raw.shape[0])):
                raw_p[t] = bool((raw[t] > 0).sum())
        # v5 gate (base)
        c5 = out.get('e_logit')
        base_p = [False] * n
        if c5 is not None:
            c5 = c5.cpu().numpy()[:n]
            for t in range(n):
                base_p[t] = bool(raw_p[t] and c5[t] > args.thr)
        # v6 gate (multi-threshold)
        v6_p = {}
        for th in [0.4, 0.5, 0.6, 0.7, 0.8]:
            v6_p[th] = [False] * n
        if vf is not None and mc is not None and lg is not None:
            vf = vf[:n].float().cpu(); mc = mc[:n].float().cpu()
            geom = geom[:n].float().cpu() if geom is not None else torch.zeros(n, 3)
            first_p = next((t for t in range(n) if raw_p[t]), 0)
            amc = mc[first_p:first_p+1]; av = vf[first_p:first_p+1]
            dev = f'cuda:{rank}'
            with torch.no_grad():
                logit = v6(vf.unsqueeze(0).to(dev), mc.unsqueeze(0).to(dev),
                           geom.unsqueeze(0).to(dev), amc.unsqueeze(0).to(dev),
                           av.unsqueeze(0).to(dev), lg.unsqueeze(0).to(dev))  # [1,T]
            e6 = logit.sigmoid().squeeze(0).cpu().numpy()[:n]
            for th in v6_p:
                for t in range(n):
                    v6_p[th][t] = bool(raw_p[t] and e6[t] > th)
        else:
            for th in v6_p:
                v6_p[th] = list(base_p)
        results.append({'category': c['category'], 'video_id': vid, 'query': c['query'][:120],
                        'n_frames': n, 'raw_presence': raw_p,
                        'base_presence': base_p, 'v6_presence': v6_p,
                        'expected_presence': c['presence'][:n]})
        if (ci + 1) % 30 == 0:
            print(f'[r{rank}][{ci+1}/{len(my_cases)}] {time.time()-t0:.0f}s', flush=True)

    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    all_res = []
    for g in gathered: all_res.extend(g)
    results = all_res

    cats = ['temporal_absence', 'global_absence', 'counterfactual_swap', 'identity_swap']
    def _getp(r, key):
        v = r[key]
        if isinstance(v, dict):
            # v6_presence is {thr: [bool...]}
            return v
        return v

    def metrics(rset, key, thr=None):
        def pres(r):
            v = r[key]
            if isinstance(v, dict) and thr is not None:
                return v[thr]
            if isinstance(v, dict):
                return v[0.5]
            return v
        ad = sum(r['n_frames'] - sum(r['expected_presence']) for r in rset)
        pd = sum(sum(r['expected_presence']) for r in rset)
        ah = sum(1 for r in rset for t in range(r['n_frames'])
                 if not r['expected_presence'][t] and pres(r)[t])
        pm = sum(1 for r in rset for t in range(r['n_frames'])
                 if r['expected_presence'][t] and not pres(r)[t])
        nc = sum(1 for r in rset for t in range(r['n_frames'])
                 if pres(r)[t] == r['expected_presence'][t])
        nt = sum(r['n_frames'] for r in rset)
        return dict(n=len(rset), absent_halluc=(ah/ad if ad else None),
                    present_miss=(pm/pd if pd else None), frame_acc=(nc/nt if nt else None))
    summary = {}
    for key in ['base_presence'] + [f'v6_t{th}' for th in [0.4, 0.5, 0.6, 0.7, 0.8]]:
        summary[key] = {}
        for cat in cats + ['overall']:
            rset = [r for r in results if cat == 'overall' or r['category'] == cat]
            if rset:
                if key == 'base_presence':
                    summary[key][cat] = metrics(rset, 'base_presence')
                else:
                    th = float(key.split('_t')[1])
                    summary[key][cat] = metrics(rset, 'v6_presence', thr=th)
    if rank == 0:
        print('=' * 60)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.save:
            json.dump({'summary': summary, 'results': results}, open(args.save, 'w'), ensure_ascii=False)
            print('saved to', args.save)

if __name__ == '__main__':
    main()

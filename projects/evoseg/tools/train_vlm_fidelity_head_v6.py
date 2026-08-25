"""Train v6 fidelity head: first-present-frame anchor + explicit anchor diffs.

Inputs per frame:
  vlm_feat [2560]->proj[256]
  mask_cond [256]
  dmc = mask_cond - anchor_mc            [256]
  dv  = vlm_proj(vlm) - vlm_proj(anchor) [256]
  geom [3]
  lang [256]
fc_in: 256+256+256+256+3+256 = 1283 -> GRU -> e_t
anchor = first PRESENT frame (fixed: frame-0 anchor was wrong on ~49% temporal).
"""
import argparse, glob, os, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
        av = self.vlm_proj(anchor_vlm)                 # [B,1,256]
        dv = v - av.expand(B, T, -1)
        dmc = mask_cond - anchor_mc.expand(B, T, -1)
        x = torch.cat([v, mask_cond, dmc, dv, geom, lang.expand(B, T, -1)], dim=-1)
        x = self.fc_in(x)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)

class DS(Dataset):
    def __init__(self, feat_dir, max_T=48):
        self.items = []
        mc = {}; geom = {}; vlm = {}
        for fp in sorted(glob.glob(os.path.join(feat_dir, 'maskcond_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                mc[(r['vid'], r['query'])] = r['mask_cond'].float()
        for fp in sorted(glob.glob(os.path.join(feat_dir, 'geom_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                geom[(r['vid'], r['query'])] = r
        for fp in sorted(glob.glob(os.path.join(feat_dir, 'vlmfeat_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                vlm[(r['vid'], r['query'])] = (r['vlm_feat'].float(), r['lang'].float())
        for fp in sorted(glob.glob(os.path.join(feat_dir, 'et_features_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                if not r.get('seg'):
                    continue
                key = (r['vid'], r['query'][:40])
                if key not in mc or key not in vlm:
                    continue
                vf, lg = vlm[key]; mc_v = mc[key]; gv = geom.get(key)
                pres = torch.tensor(r['presence'], dtype=torch.float32)
                n = min(vf.shape[0], mc_v.shape[0], pres.shape[0])
                ge = torch.stack([gv['area'], gv['cx'], gv['cy']], dim=1).float() if gv is not None else torch.zeros(n, 3)
                n = min(n, ge.shape[0])
                # first present frame anchor
                fp_idx = next((t for t in range(n) if pres[t] > 0.5), 0)
                anchor_mc = mc_v[fp_idx:fp_idx+1]
                anchor_vlm = vf[fp_idx:fp_idx+1]
                vf, mc_v, ge, pres = vf[:n], mc_v[:n], ge[:n], pres[:n]
                if n > max_T:
                    self.items.append((vf[:max_T], mc_v[:max_T], ge[:max_T], anchor_mc, anchor_vlm, lg, pres[:max_T]))
                else:
                    self.items.append((vf, mc_v, ge, anchor_mc, anchor_vlm, lg, pres))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        v, m, g, amc, av, l, p = self.items[i]
        return v, m, g, amc, av, l, p

def collate(batch):
    vs, mcs, geoms, amcs, avs, langs, press = [], [], [], [], [], [], []
    for v, m, g, amc, av, l, p in batch:
        vs.append(v); mcs.append(m); geoms.append(g); amcs.append(amc); avs.append(av); langs.append(l); press.append(p)
    maxT = max(x.shape[0] for x in vs)
    V = torch.zeros(len(batch), maxT, vs[0].shape[1]); M = torch.zeros(len(batch), maxT, mcs[0].shape[1])
    G = torch.zeros(len(batch), maxT, 3); P = torch.zeros(len(batch), maxT)
    AM = torch.zeros(len(batch), 1, mcs[0].shape[1]); AV = torch.zeros(len(batch), 1, vs[0].shape[1])
    L = torch.zeros(len(batch), 1, langs[0].shape[1]); mask = torch.zeros(len(batch), maxT, dtype=torch.bool)
    for i, (v, m, g, amc, av, l, p) in enumerate(batch):
        T = v.shape[0]
        V[i,:T] = v; M[i,:T] = m; G[i,:T] = g; P[i,:T] = p; AM[i] = amc; AV[i] = av; L[i] = l; mask[i,:T] = True
    return V, M, G, AM, AV, L, P, mask

def main():
    import torch.distributed as dist
    _rank = int(os.environ.get('LOCAL_RANK', '0')); _world = int(os.environ.get('WORLD_SIZE', '1'))
    if _world > 1:
        dist.init_process_group('nccl', init_method='env://'); torch.cuda.set_device(_rank)
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir', default='/9950backfile/chenjiahui/evo_artifacts/data/et_features')
    ap.add_argument('--out', default='/tmp/v6_head.pt')
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=8e-4)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.03); ap.add_argument('--pos-weight', type=float, default=1.5)
    ap.add_argument('--thr', type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    ds = DS(args.feat_dir)
    if _rank == 0: print(f'total v6 cases: {len(ds)}', flush=True)
    if len(ds) == 0: print('no data'); return
    n = len(ds); idx = list(range(n)); random.shuffle(idx)
    n_val = int(n * args.val_frac); val_idx = set(idx[:n_val]); train_idx = idx[n_val:]
    tr = torch.utils.data.Subset(ds, train_idx); va = torch.utils.data.Subset(ds, list(val_idx))
    tr_sampler = torch.utils.data.distributed.DistributedSampler(tr, shuffle=True) if _world > 1 else None
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=(tr_sampler is None), sampler=tr_sampler, collate_fn=collate)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)
    device = f'cuda:{_rank}' if torch.cuda.is_available() else 'cpu'
    model = V6Head().to(device)
    if _world > 1: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight).to(device))
    if _rank == 0: print(f'v6 head params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    def evaluate(dl):
        model.eval(); corr = tot = tp = fp = fn = 0
        with torch.no_grad():
            for V, M, G, AM, AV, L, P, msk in dl:
                V, M, G, AM, AV, L, P, msk = (V.to(device), M.to(device), G.to(device), AM.to(device),
                                              AV.to(device), L.to(device), P.to(device), msk.to(device))
                logit = model(V, M, G, AM, AV, L)
                pred = logit.sigmoid() > 0.5
                corr += ((pred == P.bool()) & msk).sum().item(); tot += msk.sum().item()
                tp += ((pred & P.bool()) & msk).sum().item(); fp += ((pred & ~P.bool()) & msk).sum().item()
                fn += ((~pred & P.bool()) & msk).sum().item()
        acc = corr/max(tot,1); prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        return acc, prec, rec, 2*prec*rec/max(prec+rec,1e-9)
    best = 0
    for ep in range(args.epochs):
        if tr_sampler is not None: tr_sampler.set_epoch(ep)
        model.train(); t0 = time.time(); tl = 0; nb = 0
        for V, M, G, AM, AV, L, P, msk in tr_dl:
            V, M, G, AM, AV, L, P, msk = (V.to(device), M.to(device), G.to(device), AM.to(device),
                                          AV.to(device), L.to(device), P.to(device), msk.to(device))
            logit = model(V, M, G, AM, AV, L)
            loss = loss_f(logit[msk], P[msk])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        acc, prec, rec, f1 = evaluate(va_dl)
        if _rank == 0:
            print(f'[ep{ep}] loss={tl/max(nb,1):.4f} val acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} ({time.time()-t0:.0f}s)', flush=True)
        if _world > 1: dist.barrier()
        if f1 > best and _rank == 0:
            best = f1
            torch.save({'state_dict': model.module.state_dict() if _world > 1 else model.state_dict(),
                        'config': {'vlm_dim': 2560, 'mask_dim': 256, 'lang_dim': 256, 'hidden': 512,
                                   'n_layers': 1, 'bidirectional': True, 'vlm_feat': True, 'v6': True,
                                   'thr': args.thr},
                        'stats': {'val_acc': acc, 'val_f1': f1}}, args.out)
            print(f'  saved -> {args.out}', flush=True)

if __name__ == '__main__':
    main()

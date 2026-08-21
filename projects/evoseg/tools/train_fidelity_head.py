"""Train the lightweight fidelity evaluator (e_t) for EvoSeg.

First-principles design: e_t = "is the propagated mask still faithful to the
query at frame t". Inputs per frame:
  * feat_pool [256] : global SAM2 scene feature
  * mask_cond [256] : SAM2 features pooled under the propagated mask
                      (the direct evidence of what SAM2 is tracking)
  * lang     [256]  : VLM [SEG] vector (query semantics)
The head is a small bidirectional GRU over frames -> per-frame e_t (BCE on
GT presence). Everything else is frozen; only the head is trained.

Usage:
  python train_fidelity_head.py \
      --feat-dir .../et_features --out .../temporal_existence_head.pt
"""
import argparse
import glob
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FidelityHead(nn.Module):
    def __init__(self, feat_dim=256, mask_dim=256, lang_dim=256,
                 hidden=512, n_layers=1):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(feat_dim + mask_dim * 2 + 3 + lang_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers,
                          batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, feat, mask_cond, geom, anchor, lang):
        # feat [B,T,C] mask_cond [B,T,C] geom [B,T,3] anchor [B,1,C] lang [B,1,C]
        B, T, C = feat.shape
        x = torch.cat([feat, mask_cond, geom, anchor.expand(B, T, -1),
                       lang.expand(B, T, -1)], dim=-1)  # [B,T,4C+3]
        x = self.fc_in(x)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)                # [B,T]


class ETDataset(Dataset):
    def __init__(self, feat_dir, mask_dir, max_T=48):
        self.items = []
        # index mask_cond by (vid, query)
        mc = {}
        for fp in sorted(glob.glob(os.path.join(mask_dir, 'maskcond_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                mc[(r['vid'], r['query'])] = r['mask_cond'].float()
        # index geometry by (vid, query)
        geom = {}
        for fp in sorted(glob.glob(os.path.join(mask_dir, 'geom_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                geom[(r['vid'], r['query'])] = r
        for fp in sorted(glob.glob(os.path.join(feat_dir, 'et_features_rank*.pt'))):
            for r in torch.load(fp, map_location='cpu'):
                if not r.get('seg'):
                    continue
                key = (r['vid'], r['query'][:40])
                if key not in mc:
                    continue
                feat = r['feat'].float()
                lang = r['lang'].float()
                mc_v = mc[key]
                gv = geom.get(key)
                pres = torch.tensor(r['presence'], dtype=torch.float32)
                n = min(feat.shape[0], mc_v.shape[0], pres.shape[0])
                anchor = mc_v[0:1]
                if gv is not None:
                    ge = torch.stack([gv['area'], gv['cx'], gv['cy']], dim=1).float()  # [T,3]
                    n = min(n, ge.shape[0])
                else:
                    ge = torch.zeros(n, 3)
                if n > max_T:
                    sel = feat[:max_T], mc_v[:max_T], pres[:max_T]
                    self.items.append((sel[0], sel[1], ge[:max_T], anchor, lang, sel[2]))
                    continue
                self.items.append((feat[:n], mc_v[:n], ge[:n], anchor, lang, pres[:n]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, m, g, a, l, p = self.items[i]
        return f, m, g, a, l, p


def collate(batch):
    feats, mcs, geoms, anchors, langs, press = [], [], [], [], [], []
    for f, m, g, a, l, p in batch:
        feats.append(f); mcs.append(m); geoms.append(g); anchors.append(a); langs.append(l); press.append(p)
    maxT = max(x.shape[0] for x in feats)
    C = feats[0].shape[1]
    F = torch.zeros(len(batch), maxT, C)
    M = torch.zeros(len(batch), maxT, C)
    G = torch.zeros(len(batch), maxT, 3)
    A = torch.zeros(len(batch), 1, C)
    P = torch.zeros(len(batch), maxT)
    L = torch.zeros(len(batch), 1, langs[0].shape[1])
    mask = torch.zeros(len(batch), maxT, dtype=torch.bool)
    for i, (f, m, g, a, l, p) in enumerate(batch):
        T = f.shape[0]
        F[i, :T] = f; M[i, :T] = m; G[i, :T] = g; A[i] = a; P[i, :T] = p; L[i] = l; mask[i, :T] = True
    return F, M, G, A, L, P, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir', default='/9950backfile/chenjiahui/evo_artifacts/data/et_features')
    ap.add_argument('--out', default='/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG/temporal_existence_head.pt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.03)
    ap.add_argument('--pos-weight', type=float, default=1.0,
                    help='BCE positive-class weight (>1 penalizes stopping '
                         'on present frames, reducing present_miss)')
    ap.add_argument('--thr', type=float, default=0.5,
                    help='inference sigmoid threshold stored in the ckpt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)

    ds = ETDataset(args.feat_dir, args.feat_dir)  # maskcond in same dir
    print(f'total [SEG]+maskcond cases: {len(ds)}', flush=True)
    if len(ds) == 0:
        print('no data yet'); return

    n = len(ds); idx = list(range(n)); random.shuffle(idx)
    n_val = int(n * args.val_frac)
    val_idx = set(idx[:n_val]); train_idx = idx[n_val:]
    tr = torch.utils.data.Subset(ds, train_idx)
    va = torch.utils.data.Subset(ds, list(val_idx))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FidelityHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight).to(device))
    print(f'head params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)

    def evaluate(dl):
        model.eval()
        corr = tot = tp = fp = fn = 0
        with torch.no_grad():
            for F, M, G, A, L, P, msk in dl:
                F, M, G, A, L, P, msk = (F.to(device), M.to(device), G.to(device), A.to(device),
                                         L.to(device), P.to(device), msk.to(device))
                logit = model(F, M, G, A, L)
                pred = logit.sigmoid() > 0.5
                mm = msk
                corr += ((pred == P.bool()) & mm).sum().item(); tot += mm.sum().item()
                tp += ((pred & P.bool()) & mm).sum().item()
                fp += ((pred & ~P.bool()) & mm).sum().item()
                fn += ((~pred & P.bool()) & mm).sum().item()
        acc = corr/max(tot,1); prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        return acc, prec, rec, 2*prec*rec/max(prec+rec,1e-9)

    best = 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0; nb = 0
        for F, M, G, A, L, P, msk in tr_dl:
            F, M, G, A, L, P, msk = (F.to(device), M.to(device), G.to(device), A.to(device),
                                     L.to(device), P.to(device), msk.to(device))
            logit = model(F, M, G, A, L)
            loss = loss_f(logit[msk], P[msk])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        acc, prec, rec, f1 = evaluate(va_dl)
        print(f'[ep{ep}] loss={tl/max(nb,1):.4f} val acc={acc:.4f} prec={prec:.4f} '
              f'rec={rec:.4f} f1={f1:.4f} ({time.time()-t0:.0f}s)', flush=True)
        if f1 > best:
            best = f1
            torch.save({'state_dict': model.state_dict(),
                        'config': {'feat_dim': 256, 'mask_dim': 256, 'lang_dim': 256,
                                   'hidden': 512, 'n_layers': 1, 'bidirectional': True,
                                   'anchor': True, 'thr': args.thr},
                        'stats': {'val_acc': acc, 'val_f1': f1}}, args.out)
            print(f'  saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()

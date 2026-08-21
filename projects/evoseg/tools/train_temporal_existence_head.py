"""Train a lightweight temporal existence head (e_t) for EvoSeg.

Input : per-frame SAM2 pooled features [T, 256] + [SEG] lang vector [1, 256]
        extracted by tools/extract_et_features.py, plus per-frame GT
        presence labels.
Model : small GRU (single layer, hidden 512) that scans the frame features
        in order and predicts e_t (referent exists at frame t). Temporal
        context lets it detect appearance / disappearance / re-appearance,
        which the per-frame MLP of TEG cannot.
Loss  : BCE per frame (pos_weight balances the class skew).
Train : frozen features, only the head is trainable (a few hundred K params).
Out   : temporal_existence_head.pt  {state_dict, config, train_stats}

Usage:
  python projects/evoseg/tools/train_temporal_existence_head.py \
      --feat-dir /9950backfile/chenjiahui/evo_artifacts/data/et_features \
      --out /9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG/temporal_existence_head.pt
"""
import argparse
import collections
import glob
import json
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ETHead(nn.Module):
    def __init__(self, feat_dim=256, lang_dim=256, hidden=512, n_layers=1):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(feat_dim + lang_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers,
                          batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, feat, lang):
        # feat: [B, T, C]  lang: [B, 1, C]
        B, T, C = feat.shape
        x = torch.cat([feat, lang.expand(B, T, -1)], dim=-1)   # [B,T,2C]
        x = self.fc_in(x)                                       # [B,T,H]
        out, _ = self.gru(x)                                    # [B,T,H]
        return self.head(out).squeeze(-1)                       # [B,T]


class ETDataset(Dataset):
    def __init__(self, feat_files, max_T=48):
        self.items = []
        for fp in feat_files:
            data = torch.load(fp, map_location='cpu')
            for r in data:
                if not r['seg']:
                    continue
                feat = r['feat'].float()          # [T,256]
                lang = r['lang'].float()          # [1,256]
                pres = torch.tensor(r['presence'], dtype=torch.float32)
                if feat.shape[0] != pres.shape[0]:
                    n = min(feat.shape[0], pres.shape[0])
                    feat, pres = feat[:n], pres[:n]
                if feat.shape[0] > max_T:
                    # keep the most informative window: start + around transitions
                    # simple: take head/middle/tail windows and pick the one with
                    # most presence transitions; fallback head
                    sel = feat[:max_T], pres[:max_T]
                    self.items.append((sel[0], lang, sel[1]))
                    continue
                self.items.append((feat, lang, pres))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        feat, lang, pres = self.items[i]
        return feat, lang, pres


def collate(batch):
    feats, langs, press = [], [], []
    for f, l, p in batch:
        feats.append(f); langs.append(l); press.append(p)
    maxT = max(x.shape[0] for x in feats)
    F = torch.zeros(len(batch), maxT, feats[0].shape[1])
    P = torch.zeros(len(batch), maxT)
    L = torch.zeros(len(batch), 1, langs[0].shape[1])
    mask = torch.zeros(len(batch), maxT, dtype=torch.bool)
    for i, (f, l, p) in enumerate(batch):
        T = f.shape[0]
        F[i, :T] = f; P[i, :T] = p; L[i] = l; mask[i, :T] = True
    return F, L, P, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir',
                    default='/9950backfile/chenjiahui/evo_artifacts/data/et_features')
    ap.add_argument('--out',
                    default='/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG/temporal_existence_head.pt')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.05)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.feat_dir, 'et_features_rank*.pt')))
    print(f'feature files: {files}')
    ds = ETDataset(files)
    print(f'total [SEG] training cases: {len(ds)}')

    # split
    n = len(ds)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = int(n * args.val_frac)
    val_idx = set(idx[:n_val])
    train_idx = idx[n_val:]
    tr = torch.utils.data.Subset(ds, train_idx)
    va = torch.utils.data.Subset(ds, list(val_idx))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       collate_fn=collate, num_workers=0, drop_last=True)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False,
                       collate_fn=collate, num_workers=0)

    # class balance
    npos = nneg = 0
    for f, l, p in ds.items:
        npos += p.sum().item(); nneg += (1 - p).sum().item()
    pos_w = 1.0   # balanced; nneg/npos=0.19 collapsed the model to all-absent
    print(f'present frames={int(npos)} absent frames={int(nneg)} pos_weight={pos_w:.2f}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ETHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).to(device))

    nparam = sum(p.numel() for p in model.parameters())
    print(f'head params: {nparam/1e6:.2f}M')

    def evaluate(dl):
        model.eval()
        tot = corr = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for F, L, P, M in dl:
                F, L, P, M = F.to(device), L.to(device), P.to(device), M.to(device)
                logit = model(F, L)
                pred = (logit.sigmoid() > 0.5)
                m = M
                corr += ((pred == P.bool()) & m).sum().item()
                tot += m.sum().item()
                tp += ((pred & P.bool()) & m).sum().item()
                fp += ((pred & ~P.bool()) & m).sum().item()
                fn += ((~pred & P.bool()) & m).sum().item()
        acc = corr / max(tot, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return acc, prec, rec, f1

    best_f1 = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot_loss = 0; nb = 0
        for F, L, P, M in tr_dl:
            F, L, P, M = F.to(device), L.to(device), P.to(device), M.to(device)
            logit = model(F, L)
            loss = loss_f(logit[M], P[M])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item(); nb += 1
        acc, prec, rec, f1 = evaluate(va_dl)
        print(f'[ep{ep}] loss={tot_loss/nb:.4f} val acc={acc:.4f} '
              f'prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'state_dict': model.state_dict(),
                        'config': {'feat_dim': 256, 'lang_dim': 256,
                                   'hidden': 512, 'n_layers': 1,
                                   'pos_weight': pos_w},
                        'stats': {'val_acc': acc, 'val_f1': f1}},
                       args.out)
            print(f'  saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()

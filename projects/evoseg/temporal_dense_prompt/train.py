"""Train parameter-matched static and temporal TDSP heads on official train data."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from projects.evoseg.temporal_dense_prompt.protocol import DensePromptHead, STAGES, count_parameters, identity, resize_mask, stage_endpoints, state_index, training_loss, video_fold


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def samples(manifest: dict, temporal_records: Path, static_records: Path, features: Path):
    temporal, static = state_index(temporal_records), state_index(static_records)
    values=[]
    for item in manifest['objects']:
        endpoints=stage_endpoints(item['frame_count'])
        for expression in item['expressions']:
            key=identity(item,expression)
            if key not in temporal or key not in static: raise RuntimeError(f'missing states: {key}')
            for stage, endpoint in zip(STAGES,endpoints):
                feature=features/f"{item['video_id']}__{endpoint}.npz"
                mask=Path(manifest['dataset']['annotation_root'])/item['video_id']/str(item['object_id'])/f"{item['frame_names'][endpoint]}.png"
                if not feature.is_file() or not mask.is_file(): raise RuntimeError(f'missing feature/mask: {feature} {mask}')
                values.append({'identity':key,'video_id':item['video_id'],'stage':stage,'temporal':temporal[key],'static':static[key],'feature':feature,'mask':mask,'fold':video_fold(item['video_id'])})
    if not {x['fold'] for x in values} == {'train','internal_val'}: raise RuntimeError('fixed split is empty')
    return values


def load_batch(rows, kind, device):
    states=[]; features=[]; masks=[]
    for row in rows:
        state_path = row['temporal'] if kind == 'temporal' or row['stage'] == 7 else row['static']
        with np.load(state_path) as z: states.append(np.asarray(z[f'{kind}_{row["stage"]}_z'],dtype=np.float32))
        feature=np.load(row['feature'])['feature'].astype(np.float32); features.append(feature)
        masks.append(resize_mask(row['mask'],feature.shape[-2:]).numpy())
    return torch.from_numpy(np.stack(states)).to(device),torch.from_numpy(np.stack(features)).to(device),torch.from_numpy(np.stack(masks)).to(device)


def evaluate(model, rows, kind, device, batch_size):
    model.eval(); losses=[]
    with torch.inference_mode():
        for offset in range(0,len(rows),batch_size):
            state,feature,target=load_batch(rows[offset:offset+batch_size],kind,device); losses.append(float(training_loss(model(state,feature),target))*len(state))
    return sum(losses)/len(rows)


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device); device=f'cuda:{args.device}'; output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text()); rows=samples(manifest,Path(args.temporal_records),Path(args.static_records),Path(args.features)); train=[x for x in rows if x['fold']=='train']; val=[x for x in rows if x['fold']=='internal_val']
    static=DensePromptHead().to(device); temporal=DensePromptHead().to(device); temporal.load_state_dict(static.state_dict())
    if count_parameters(static)>=2_000_000 or count_parameters(static)!=count_parameters(temporal): raise RuntimeError('parameter protocol failed')
    optimizers=[torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4) for model in (static,temporal)]
    curves=[]; best=float('inf'); best_epoch=-1; patience=0; started=time.monotonic(); torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(args.max_epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(len(train)); totals={'static':0.0,'temporal':0.0}
        static.train(); temporal.train()
        for offset in range(0,len(order),args.batch_size):
            batch=[train[i] for i in order[offset:offset+args.batch_size]]
            for kind,model,optimizer in [('static',static,optimizers[0]),('temporal',temporal,optimizers[1])]:
                state,feature,target=load_batch(batch,kind,device); optimizer.zero_grad(set_to_none=True); loss=training_loss(model(state,feature),target); loss.backward(); optimizer.step(); totals[kind]+=float(loss.detach())*len(batch)
        val_static=evaluate(static,val,'static',device,args.batch_size); val_temporal=evaluate(temporal,val,'temporal',device,args.batch_size); joint=(val_static+val_temporal)/2
        curve={'epoch':epoch+1,'train_static':totals['static']/len(train),'train_temporal':totals['temporal']/len(train),'val_static':val_static,'val_temporal':val_temporal,'joint_val':joint}; curves.append(curve); print(curve,flush=True)
        if joint < best-args.min_delta:
            best=joint; best_epoch=epoch+1; patience=0; torch.save({'static':static.state_dict(),'temporal':temporal.state_dict(),'epoch':best_epoch,'seed':args.seed},output/'checkpoint.pt')
        else: patience+=1
        atomic(output/'training_status.json',{'state':'running','pid':__import__('os').getpid(),'seed':args.seed,'epoch':epoch+1,'best_epoch':best_epoch,'best_joint_val':best,'curve':curves})
        if patience>=args.patience: break
    status={'state':'complete','pid':__import__('os').getpid(),'seed':args.seed,'train_samples':len(train),'internal_val_samples':len(val),'train_videos':len({x['video_id'] for x in train}),'internal_val_videos':len({x['video_id'] for x in val}),'parameter_count_each':count_parameters(static),'identical_initialization':True,'identical_updates':len(curves),'best_epoch':best_epoch,'best_joint_val':best,'epochs':len(curves),'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(args.device),'curve':curves}; atomic(output/'training_status.json',status); return 0


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True);p.add_argument('--temporal-records',required=True);p.add_argument('--static-records',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,choices=(11,23,42),required=True);p.add_argument('--device',type=int,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--max-epochs',type=int,default=40);p.add_argument('--patience',type=int,default=6);p.add_argument('--min-delta',type=float,default=1e-4); return p.parse_args()


if __name__=='__main__': raise SystemExit(run(parse_args()))

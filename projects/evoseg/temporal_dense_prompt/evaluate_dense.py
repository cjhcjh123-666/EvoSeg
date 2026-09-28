"""Evaluate TDSP heatmaps before pixel execution, including fixed state shuffle."""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from projects.evoseg.temporal_dense_prompt.protocol import DensePromptHead, STAGES, identity, stage_endpoints, state_index
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as h: h.write(json.dumps(row)+'\n');h.flush();os.fsync(h.fileno())


def load_models(paths,device):
    result=[]
    for path in paths:
        payload=torch.load(path,map_location='cpu',weights_only=True); pair={}
        for kind in ('static','temporal'):
            model=DensePromptHead().to(device);model.load_state_dict(payload[kind]);model.eval();pair[kind]=model
        result.append(pair)
    return result


def run(args):
    torch.cuda.set_device(args.device);device=f'cuda:{args.device}';manifest=json.loads(Path(args.manifest).read_text());objects=manifest['objects'][args.object_start:args.object_stop]
    objects=[x for i,x in enumerate(objects) if i%args.num_shards==args.shard_index]
    temporal=state_index(Path(args.temporal_records));static=state_index(Path(args.static_records));models=load_models(args.checkpoints,device)
    all_expressions=[(identity(item,e),item['video_id']) for item in manifest['objects'] for e in item['expressions'] if identity(item,e) in temporal]
    ordered=sorted(all_expressions); donor={}
    for offset,(key,video) in enumerate(ordered):
        for step in range(1,len(ordered)):
            candidate,cvideo=ordered[(offset+step)%len(ordered)]
            if cvideo!=video: donor[key]=candidate;break
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);records=output/f'dense_results.shard{args.shard_index}.jsonl';done=set()
    if records.is_file(): done={(r['identity'],r['condition'],r['stage']) for line in records.open() if line.strip() for r in [json.loads(line)] if r.get('status')=='success'}
    started=time.monotonic();planned=sum(len(x['expressions'])*len(STAGES)*3 for x in objects); torch.cuda.reset_peak_memory_stats(args.device)
    with torch.inference_mode():
      for item in objects:
        endpoints=stage_endpoints(item['frame_count'])
        for expression in item['expressions']:
          key=identity(item,expression)
          if key not in temporal or key not in static:
            for stage in STAGES:
              for condition in ('static','temporal','shuffled'):
                if (key,condition,stage) not in done:
                  append(records,{'status':'failed_missing_state','identity':key,'video_id':item['video_id'],'object_id':item['object_id'],'expression_id':expression['expression_id'],'description_type':expression['type'],'expression':expression['text'],'condition':condition,'stage':stage,'error':'missing_temporal_state' if key not in temporal else 'missing_static_state','gt_entered_prompt_generation':False})
            continue
          for stage,endpoint in zip(STAGES,endpoints):
            feature_path=Path(args.features)/f"{item['video_id']}__{endpoint}.npz"; feature=torch.from_numpy(np.load(feature_path)['feature'].astype(np.float32))[None].to(device)
            for condition in ('static','temporal','shuffled'):
              if (key,condition,stage) in done: continue
              state_key=donor[key] if condition=='shuffled' else key; kind='temporal' if condition!='static' else 'static'; state_path=temporal[state_key] if kind=='temporal' or stage==7 else static[state_key]
              with np.load(state_path) as z: state=torch.from_numpy(np.asarray(z[f'{kind}_{stage}_z'],dtype=np.float32))[None].to(device)
              logits=torch.stack([pair[kind](state,feature)[0] for pair in models]).mean(0);prob=logits.sigmoid();heatmap=prob.detach().cpu().numpy().astype(np.float16)
              heatpath=output/'heatmaps'/condition/f"{key.replace('/','__')}__s{stage}.npz";heatpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(heatpath,probability=heatmap)
              mask_path=Path(manifest['dataset']['annotation_root'])/item['video_id']/str(item['object_id'])/f"{item['frame_names'][endpoint]}.png"
              with Image.open(mask_path) as image: gt=np.asarray(image.convert('L'))>0
              pred=F.interpolate(prob[None,None],size=gt.shape,mode='bilinear',align_corners=False)[0,0].cpu().numpy()>=.5
              j=float(db_eval_iou(gt,pred));f=float(db_eval_boundary(gt,pred)); row={'status':'success','identity':key,'video_id':item['video_id'],'object_id':item['object_id'],'expression_id':expression['expression_id'],'description_type':expression['type'],'expression':expression['text'],'condition':condition,'stage':stage,'anchor_frame_index':endpoint,'state_identity':state_key,'shuffle_different_video':condition!='shuffled' or dict(ordered)[state_key]!=item['video_id'],'threshold':.5,'J':j,'F':f,'J_and_F':(j+f)/2,'heatmap_path':str(heatpath),'gt_entered_prompt_generation':False};append(records,row);done.add((key,condition,stage))
        status={'state':'running','pid':os.getpid(),'planned':planned,'completed':len(done),'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(args.device)};(output/f'STATUS.shard{args.shard_index}.json').write_text(json.dumps(status,indent=2)+'\n')
    status.update(state='complete');(output/f'STATUS.shard{args.shard_index}.json').write_text(json.dumps(status,indent=2)+'\n');return 0


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--temporal-records',required=True);p.add_argument('--static-records',required=True);p.add_argument('--features',required=True);p.add_argument('--checkpoints',nargs=3,required=True);p.add_argument('--output',required=True);p.add_argument('--device',type=int,required=True);p.add_argument('--object-start',type=int,default=0);p.add_argument('--object-stop',type=int,default=274);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);return p.parse_args()


if __name__=='__main__':raise SystemExit(run(parse_args()))

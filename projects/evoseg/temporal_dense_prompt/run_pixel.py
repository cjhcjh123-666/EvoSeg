"""Execute TDSP prompts through the official public SAM3.1 point API."""
from __future__ import annotations
import argparse, inspect, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from projects.evoseg.temporal_compiler.sam31_candidate_protocol import audit_loaded_checkpoint, sha256, start_multiplex_session
from projects.evoseg.temporal_dense_prompt.protocol import STAGES, identity, local_extrema_points, one_point, stage_endpoints
from projects.evoseg.temporal_grounding_interface.run_dynamic_interface import selected_object_mask, stream_masks
from projects.evoseg.temporal_seg.official_metrics import db_eval_boundary, db_eval_iou


def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as h:h.write(json.dumps(row)+'\n');h.flush();os.fsync(h.fileno())


def encode(mask):
 value=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)));value['counts']=value['counts'].decode('ascii');return value


def heatmap_path(root,key,kind,stage): return root/'heatmaps'/kind/f"{key.replace('/','__')}__s{stage}.npz"


def build_plan(root,item,expression,condition):
 kind,form=condition.split('_');key=identity(item,expression);plan=[]
 for stage,endpoint in zip(STAGES,stage_endpoints(item['frame_count'])):
  probability=np.load(heatmap_path(root,key,kind,stage))['probability'].astype(np.float32)
  points,labels=one_point(probability) if form=='p1' else local_extrema_points(probability,4,4)
  plan.append({'stage':stage,'anchor_frame_index':endpoint,'points':points,'point_labels':labels,'heatmap_path':str(heatmap_path(root,key,kind,stage))})
 return plan


def infer(predictor,video_path,frame_count,shape,plan):
 session,compat=start_multiplex_session(predictor,str(video_path));sid=session['session_id'];obj=1;final={};maintained={};previous=-1;stages=[]
 try:
  for row in plan:
   endpoint=row['anchor_frame_index'];chunk=previous+1;pre=maintained.get(endpoint);before=time.monotonic()
   response=predictor.add_prompt(session_id=sid,frame_idx=endpoint,points=row['points'],point_labels=row['point_labels'],obj_id=obj,rel_coordinates=True)
   post=selected_object_mask(response['outputs'],obj,shape)
   backward=stream_masks(predictor,sid,'backward',endpoint,endpoint-chunk+1,obj,shape);backward[endpoint]=post
   forward=stream_masks(predictor,sid,'forward',endpoint,frame_count-endpoint,obj,shape);maintained.update(backward);maintained.update(forward);maintained[endpoint]=post
   for index in range(chunk,endpoint+1):final[index]=backward.get(index,maintained.get(index,np.zeros(shape,bool)))
   stages.append({**row,'pre_update_area':None if pre is None else int(pre.sum()),'post_update_mask':post,'update_latency_seconds':time.monotonic()-before});previous=endpoint
 finally: predictor.close_session(sid)
 if len(final)!=frame_count: raise RuntimeError(f'incomplete track {len(final)}/{frame_count}')
 return final,stages,compat


def gt_mask(path,shape):
 if not Path(path).is_file():return np.zeros(shape,dtype=bool)
 with Image.open(path) as image:return np.asarray(image.convert('L'))>0


def run(args):
 repo=Path(args.sam3_repo).resolve();sys.path.insert(0,str(repo));from sam3.model_builder import build_sam3_multiplex_video_predictor
 source=Path(inspect.getfile(build_sam3_multiplex_video_predictor)).resolve();checkpoint=Path(args.checkpoint).resolve()
 if repo not in source.parents or sha256(checkpoint)!=args.expected_checkpoint_sha256:raise RuntimeError('official runtime/checkpoint audit failed')
 manifest=json.loads(Path(args.manifest).read_text());selected=manifest['objects'][args.object_start:args.object_stop];selected=[x for i,x in enumerate(selected) if i%args.num_shards==args.shard_index]
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True);records=output/f'pixel_results.shard{args.shard_index}.jsonl';done=set()
 if records.is_file():done={(r['identity'],r['condition']) for line in records.open() if line.strip() for r in [json.loads(line)] if r.get('status')=='success'}
 torch.cuda.set_device(args.device);predictor=build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint),max_num_objects=16,multiplex_count=16,use_fa3=False,compile=False,warm_up=False,async_loading_frames=False);audit=audit_loaded_checkpoint(predictor.model,checkpoint);started=time.monotonic();planned=sum(len(x['expressions'])*4 for x in selected);failed=0
 for item in selected:
  video_path=Path(manifest['dataset']['image_root'])/item['video_id'];
  with Image.open(video_path/f"{item['frame_names'][0]}.jpg") as image:shape=(image.height,image.width)
  for expression in item['expressions']:
   key=identity(item,expression)
   for condition in ('static_p1','temporal_p1','static_p8','temporal_p8'):
    if (key,condition) in done:continue
    base={'identity':key,'video_id':item['video_id'],'object_id':item['object_id'],'expression_id':expression['expression_id'],'description_type':expression['type'],'expression':expression['text'],'condition':condition,'status':'failed','gt_entered_prompt_generation':False}
    try:
     plan=build_plan(Path(args.dense_root),item,expression,condition);torch.cuda.reset_peak_memory_stats(args.device);torch.cuda.synchronize(args.device);begin=time.monotonic();masks,stages,compat=infer(predictor,video_path,item['frame_count'],shape,plan);torch.cuda.synchronize(args.device);latency=time.monotonic()-begin
     js=[];fs=[]
     for frame,path in zip(item['evaluation_frame_indices'],item['evaluation_mask_paths']):
      gt=gt_mask(path,shape);js.append(float(db_eval_iou(gt,masks[frame])));fs.append(float(db_eval_boundary(gt,masks[frame])))
     stage_metrics=[]
     for row in stages:
      endpoint=row['anchor_frame_index'];path=Path(manifest['dataset']['annotation_root'])/item['video_id']/str(item['object_id'])/f"{item['frame_names'][endpoint]}.png";gt=gt_mask(path,shape);prob=np.load(row['heatmap_path'])['probability'].astype(np.float32);dense=np.asarray(Image.fromarray((prob*255).astype(np.uint8)).resize((shape[1],shape[0]),Image.Resampling.BILINEAR))>=128
      dj=float(db_eval_iou(gt,dense));df=float(db_eval_boundary(gt,dense));ij=float(db_eval_iou(gt,row['post_update_mask']));iff=float(db_eval_boundary(gt,row['post_update_mask']));stage_metrics.append({'stage':row['stage'],'anchor_frame_index':endpoint,'dense_J_and_F':(dj+df)/2,'init_J_and_F':(ij+iff)/2,'dense_to_init_loss_J_and_F':(dj+df-ij-iff)/2,'positive_count':sum(x==1 for x in row['point_labels']),'negative_count':sum(x==0 for x in row['point_labels']),'update_latency_seconds':row['update_latency_seconds']})
     mask_path=output/'masks'/condition/f"{key.replace('/','__')}.json";mask_path.parent.mkdir(parents=True,exist_ok=True);mask_path.write_text(json.dumps({'evaluation_frame_indices':item['evaluation_frame_indices'],'masks':[encode(masks[i]) for i in item['evaluation_frame_indices']]})+'\n')
     j=float(np.mean(js));f=float(np.mean(fs));base.update(status='success',J=j,F=f,J_and_F=(j+f)/2,stage_metrics=stage_metrics,mean_dense_J_and_F=float(np.mean([x['dense_J_and_F'] for x in stage_metrics])),mean_init_J_and_F=float(np.mean([x['init_J_and_F'] for x in stage_metrics])),dense_to_init_loss_J_and_F=float(np.mean([x['dense_to_init_loss_J_and_F'] for x in stage_metrics])),init_to_final_loss_J_and_F=float(np.mean([x['init_J_and_F'] for x in stage_metrics]))-(j+f)/2,latency_seconds_synchronized=latency,peak_memory_bytes=torch.cuda.max_memory_allocated(args.device),mask_path=str(mask_path),public_api_compat=compat,plan=plan)
     done.add((key,condition))
    except Exception as error:failed+=1;base.update(error=str(error),traceback=traceback.format_exc())
    append(records,base)
  status={'state':'running','pid':os.getpid(),'planned':planned,'completed':len(done),'failed':failed,'elapsed_seconds':time.monotonic()-started,'estimated_remaining_seconds':(time.monotonic()-started)/max(len(done)+failed,1)*max(planned-len(done)-failed,0),'official_builder_source':str(source),'weight_audit':audit};(output/f'STATUS.shard{args.shard_index}.json').write_text(json.dumps(status,indent=2)+'\n')
 status.update(state='complete' if failed==0 else 'complete_with_failures');(output/f'STATUS.shard{args.shard_index}.json').write_text(json.dumps(status,indent=2)+'\n');return 0


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--dense-root',required=True);p.add_argument('--output',required=True);p.add_argument('--sam3-repo',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--expected-checkpoint-sha256',required=True);p.add_argument('--device',type=int,required=True);p.add_argument('--object-start',type=int,default=0);p.add_argument('--object-stop',type=int,default=274);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);return p.parse_args()


if __name__=='__main__':raise SystemExit(run(parse_args()))

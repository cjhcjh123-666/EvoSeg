"""Two-video real-data smoke for frozen features, states, loss and shuffle."""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from projects.evoseg.temporal_dense_prompt.protocol import DensePromptHead, STAGES, identity, resize_mask, stage_endpoints, state_index, training_loss


def run(args):
 torch.cuda.set_device(args.device);device=f'cuda:{args.device}';manifest=json.loads(Path(args.manifest).read_text());videos=[];items=[]
 for item in manifest['objects']:
  if item['video_id'] not in videos:videos.append(item['video_id'])
  if item['video_id'] in videos[:2]:items.append(item)
  if len(videos)>=3:break
 temporal=state_index(Path(args.temporal_records));static=state_index(Path(args.static_records));samples=[]
 for item in items:
  endpoint=stage_endpoints(item['frame_count'])[0];stage=STAGES[0]
  feature=np.load(Path(args.features)/f"{item['video_id']}__{endpoint}.npz")['feature'].astype(np.float32)
  mask=Path(manifest['dataset']['annotation_root'])/item['video_id']/str(item['object_id'])/f"{item['frame_names'][endpoint]}.png";target=resize_mask(mask,feature.shape[-2:]).numpy()
  for expression in item['expressions'][:2]:
   key=identity(item,expression)
   with np.load(temporal[key]) as z:t=np.asarray(z[f'temporal_{stage}_z'],np.float32)
   with np.load(static[key]) as z:s=np.asarray(z[f'static_{stage}_z'],np.float32)
   samples.append((t,s,feature,target,item['video_id']))
 t,s,f,y,_=zip(*samples);t=torch.from_numpy(np.stack(t)).to(device);s=torch.from_numpy(np.stack(s)).to(device);f=torch.from_numpy(np.stack(f)).to(device);y=torch.from_numpy(np.stack(y)).to(device)
 model=DensePromptHead().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3);losses=[]
 for _ in range(12):optimizer.zero_grad();loss=training_loss(model(t,f),y);loss.backward();optimizer.step();losses.append(float(loss.detach()))
 with torch.inference_mode():normal=model(t,f).sigmoid();shuffled=model(torch.roll(t,1,0),f).sigmoid()
 result={'status':'pass' if losses[-1]<losses[0] and float(normal.std())>.001 and not torch.allclose(normal,shuffled) else 'fail','videos':videos[:2],'samples':len(samples),'initial_loss':losses[0],'final_loss':losses[-1],'heatmap_mean':float(normal.mean()),'heatmap_std':float(normal.std()),'shuffle_mean_abs_change':float((normal-shuffled).abs().mean()),'ground_truth_used_only_for_training_loss':True}
 Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(result);return 0 if result['status']=='pass' else 2


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--temporal-records',required=True);p.add_argument('--static-records',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--device',type=int,required=True);return p.parse_args()
if __name__=='__main__':raise SystemExit(run(parse_args()))

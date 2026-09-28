"""Export lightweight TDSP CSVs, reports and figures from immutable artifacts."""
from __future__ import annotations
import argparse,csv,json,shutil
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def read_jsonl(paths):
 out=[]
 for path in paths:
  out.extend(json.loads(line) for line in path.open() if line.strip())
 return out


def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 keys=sorted({k for row in rows for k in row if k not in {'plan','stage_metrics','public_api_compat','traceback'}})
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows({k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in row.items() if k in keys} for row in rows)


def pct(value):return 'N/A' if value is None else f'{100*value:.2f}'


def decode(rle):
 value=dict(rle);value['counts']=value['counts'].encode('ascii');return mask_utils.decode(value).astype(bool)


def run(args):
 root=Path(args.run_dir);docs=Path(args.docs);docs.mkdir(parents=True,exist_ok=True);(docs/'figures/qualitative_cases').mkdir(parents=True,exist_ok=True)
 pilot=json.loads((root/'pilot64/summary/summary.json').read_text());confirm_path=root/'confirm210/summary/summary.json';confirm=json.loads(confirm_path.read_text()) if confirm_path.is_file() else None;final=confirm or pilot
 dense_paths=sorted((root/'pilot64/dense').glob('dense_results.shard*.jsonl'))+sorted((root/'confirm210/dense').glob('dense_results.shard*.jsonl'));pixel_paths=sorted((root/'pilot64/pixel').glob('pixel_results.shard*.jsonl'))+sorted((root/'confirm210/pixel').glob('pixel_results.shard*.jsonl'));dense=read_jsonl(dense_paths);pixel=read_jsonl(pixel_paths)
 write_csv(docs/'dense_prompt_results.csv',dense);write_csv(docs/'final_segmentation_results.csv',pixel)
 paired_source=(root/'confirm210/summary/paired_dynamic_results.csv') if confirm else (root/'pilot64/summary/paired_dynamic_results.csv');shutil.copy2(paired_source,docs/'paired_dynamic_results.csv')
 training=[]
 for seed in (11,23,42):training.append(json.loads((root/f'training/seed{seed}/training_status.json').read_text()))
 write_csv(docs/'runtime_summary.csv',[{'component':'training',**x} for x in training]+[{'component':'final_dynamic_temporal_p8',**final['pixel_losses']}])
 params=training[0]['parameter_count_each'];method=f"""# TDSP method prototype

Frozen Sa2VA cumulative or anchor-only state is combined with the frozen official SAM3.1 256x72x72 propagation feature map by a {params:,}-parameter dot-product dense head. K=4, threshold=0.5, and deterministic P1/P8 point conversion were fixed before validation. SAM3.1 and Sa2VA remained frozen; only official Long-RVOS train masks supervised BCE+Dice. Public `add_prompt` points update one persistent `obj_id`; no private mask-prompt API or validation tuning was used.
""";(docs/'METHOD_PROTO.md').write_text(method)
 lines=['# Training report','']+[f"- seed {x['seed']}: complete, best epoch {x['best_epoch']}, {x['epochs']} matched updates, peak {x['peak_memory_bytes']/2**30:.2f} GiB, {x['elapsed_seconds']/60:.1f} min" for x in training];lines+=['',f'- Trainable parameters per head: {params:,}.','- Static and Temporal heads used identical architecture, initialization, batches, optimizer and update count.'];(docs/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n')
 def phase_report(name,value):
  d=value['by_type']['dynamic'];return f"# {name}\n\n- Static-DSP P8 Dynamic J&F: {pct(d['static_p8'])}\n- Temporal-DSP P8 Dynamic J&F: {pct(d['temporal_p8'])}\n- Temporal - Static: {100*d['final_temporal8_minus_static8']:+.2f} pp; 95% CI [{100*d['final_temporal8_minus_static8_ci95'][0]:+.2f}, {100*d['final_temporal8_minus_static8_ci95'][1]:+.2f}] pp.\n- Dense-map Temporal - Static: {100*d['dense_temporal_minus_static']:+.2f} pp.\n- Temporal - shuffled dense map: {100*d['dense_temporal_minus_shuffled']:+.2f} pp.\n- Temporal P8 - P1: {100*d['temporal8_minus_temporal1']:+.2f} pp.\n- Gate: **{value['decision']}**.\n"
 (docs/'PILOT64_RESULTS.md').write_text(phase_report('Pilot64 results',pilot))
 if confirm:(docs/'CONFIRM210_RESULTS.md').write_text(phase_report('Confirmatory 210-object results',confirm))
 loss=final['pixel_losses'];(docs/'PIXEL_LOSS_DECOMPOSITION.md').write_text(f"# Pixel loss decomposition\n\nDynamic Temporal-DSP P8 mean dense-to-initialization loss: {100*loss['dense_to_init']:+.2f} pp. Mean initialization-to-final loss: {100*loss['init_to_final']:+.2f} pp. These signs use `upstream J&F - downstream J&F`; negative values mean the downstream stage improved.\n")
 failures=[x for x in dense+pixel if x.get('status')!='success'];(docs/'FAILURE_ANALYSIS.md').write_text(f"# Failure analysis\n\n- Retained failed records: {len(failures)}.\n- Successful dense stage records: {sum(x.get('status')=='success' for x in dense)}.\n- Successful final segmentation records: {sum(x.get('status')=='success' for x in pixel)}.\n- Qualitative selection is deterministic from Dynamic Temporal-P8 minus Static-P8 and includes success, near-zero, and damage cases.\n")
 decision=final['decision'];recommendation={'GO':'Proceed only after review toward the fixed TDSP direction.','MIXED':'Keep the frozen VLM and dense head; next isolate pixel memory/executor loss.','NO-GO':'Stop TDSP and do not add Transformer/Agent/RL capacity.','GO_TO_CONFIRM210':'Confirmatory evaluation is still required.'}[decision]
 (docs/'FINAL_GO_NOGO.md').write_text(f"# Final decision\n\n**{decision}**\n\n{recommendation}\n")
 snapshot=root/'MORNING_SNAPSHOT.md';(docs/'MORNING_SNAPSHOT.md').write_text(snapshot.read_text() if snapshot.is_file() else '# Morning snapshot\n\nRun completed before the 08:00 snapshot deadline.\n')
 # Compact main figures.
 d=final['by_type']['dynamic'];
 figures=[('dynamic_jf_main.png',['Static P8','Temporal P8'],[d['static_p8'],d['temporal_p8']]),('onepoint_vs_multipoint.png',['Temporal P1','Temporal P8'],[d['temporal_p1'],d['temporal_p8']]),('dense_grounding_static_vs_temporal.png',['Static dense','Temporal dense','Shuffled'],[d['dense_static'],d['dense_temporal'],d['dense_shuffled']]),('pixel_loss_decomposition.png',['Dense→init','Init→final'],[loss['dense_to_init'],loss['init_to_final']])]
 for filename,labels,values in figures:
  fig,ax=plt.subplots(figsize=(5,3.2));ax.bar(labels,np.asarray(values)*100);ax.set_ylabel('J&F / delta (pp)');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(docs/'figures'/filename,dpi=160);plt.close(fig)
 # Deterministic Dynamic success/no-change/damage cases on the final evaluated split.
 manifest=json.loads(Path(args.manifest).read_text());item_index={f"{x['dataset']}/{x['video_id']}/{x['object_id']}":x for x in manifest['objects']}
 by_identity=defaultdict(dict)
 for row in pixel:
  if row.get('status')=='success' and row['description_type']=='dynamic' and row['condition'] in {'static_p8','temporal_p8'}:by_identity[row['identity']][row['condition']]=row
 candidates=[]
 for key,pair in by_identity.items():
  if len(pair)==2:candidates.append((pair['temporal_p8']['J_and_F']-pair['static_p8']['J_and_F'],key,pair))
 selected=[];used=set()
 for label,ordered in [('success',sorted(candidates,reverse=True)),('no_change',sorted(candidates,key=lambda x:abs(x[0]))),('damage',sorted(candidates))]:
  count=0
  for delta,key,pair in ordered:
   if key in used:continue
   if label=='success' and delta<=0:continue
   if label=='damage' and delta>=0:continue
   used.add(key);selected.append((label,delta,key,pair));count+=1
   if count==3:break
 for label,delta,key,pair in selected:
  base='/'.join(key.split('/')[:3]);item=item_index[base];frame_index=item['evaluation_frame_indices'][-1];frame_name=item['frame_names'][frame_index]
  image=np.asarray(Image.open(Path(manifest['dataset']['image_root'])/item['video_id']/f'{frame_name}.jpg').convert('RGB'));gt=np.asarray(Image.open(item['evaluation_mask_paths'][-1]).convert('L'))>0
  masks={};heats={}
  for condition in ('static_p8','temporal_p8'):
   payload=json.loads(Path(pair[condition]['mask_path']).read_text());masks[condition]=decode(payload['masks'][-1]);stage=pair[condition]['plan'][-1];heats[condition]=np.load(stage['heatmap_path'])['probability']
  fig,axes=plt.subplots(1,6,figsize=(15,3));values=[image,gt,heats['static_p8'],heats['temporal_p8'],masks['static_p8'],masks['temporal_p8']];titles=['Frame','GT','Static heatmap','Temporal heatmap','Static final','Temporal final']
  for ax,value,title in zip(axes,values,titles):ax.imshow(value,cmap=None if value.ndim==3 else 'viridis');ax.set_title(title);ax.axis('off')
  fig.suptitle(f"{label}: delta {100*delta:+.2f} pp | {pair['temporal_p8']['expression']}",fontsize=9);fig.tight_layout();fig.savefig(docs/'figures/qualitative_cases'/f"{label}__{key.replace('/','__')}.png",dpi=130);plt.close(fig)
  return 0


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--docs',required=True);p.add_argument('--manifest',required=True);return p.parse_args()
if __name__=='__main__':raise SystemExit(run(parse_args()))

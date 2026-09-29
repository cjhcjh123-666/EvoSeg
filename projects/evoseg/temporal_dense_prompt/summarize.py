"""Object-balanced TDSP summaries, clustered bootstrap, reports and gate."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def rows(root,pattern):
 result=[]
 for path in sorted(root.glob(pattern)):
  result.extend(json.loads(line) for line in path.open() if line.strip() and json.loads(line).get('status')=='success')
 return result


def pixel_coverage(dense, pixel):
 expected_identities={row['identity'] for row in dense}
 expected={(identity,condition) for identity in expected_identities for condition in ('static_p1','temporal_p1','static_p8','temporal_p8')}
 observed={(row['identity'],row['condition']) for row in pixel}
 missing=sorted(expected-observed)
 return {
  'expected_identity_condition_pairs':len(expected),
  'successful_identity_condition_pairs':len(expected&observed),
  'missing_identity_condition_pairs':len(missing),
  'complete':not missing,
  'missing_examples':[{'identity':identity,'condition':condition} for identity,condition in missing[:20]],
 }


def object_means(values,metric,condition,kind):
 grouped=defaultdict(list)
 for row in values:
  if row['condition']==condition and row['description_type']==kind:grouped[(row['video_id'],str(row['object_id']))].append(float(row[metric]))
 return {key:float(np.mean(v)) for key,v in grouped.items()}


def delta(values,metric,a,b,kind):
 left=object_means(values,metric,a,kind);right=object_means(values,metric,b,kind);keys=sorted(set(left)&set(right));return {k:left[k]-right[k] for k in keys}


def bootstrap(deltas,seed=42,reps=2000):
 by_video=defaultdict(list)
 for (video,_),value in deltas.items():by_video[video].append(value)
 videos=sorted(by_video);rng=np.random.default_rng(seed);samples=[]
 for _ in range(reps):
  chosen=rng.choice(videos,len(videos),replace=True);samples.append(float(np.mean([x for v in chosen for x in by_video[v]])))
 return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]


def run(args):
 dense=rows(Path(args.dense_root),'dense_results.shard*.jsonl');pixel=rows(Path(args.pixel_root),'pixel_results.shard*.jsonl')
 # Dense rows have four stages: first average stages within expression.
 dg=defaultdict(list)
 for row in dense:dg[(row['identity'],row['condition'])].append(row)
 dense_expr=[]
 for (_,condition),group in dg.items():
  base=dict(group[0]);base['J_and_F']=float(np.mean([x['J_and_F'] for x in group]));dense_expr.append(base)
 coverage=pixel_coverage(dense,pixel)
 result={'phase':args.phase,'dense_success_rows':len(dense),'pixel_success_rows':len(pixel),'pixel_coverage':coverage,'by_type':{}}
 for kind in ('static','dynamic','hybrid'):
  d={}
  for condition in ('static','temporal','shuffled'):
   values=object_means(dense_expr,'J_and_F',condition,kind);d[f'dense_{condition}']=float(np.mean(list(values.values()))) if values else None
  for condition in ('static_p1','temporal_p1','static_p8','temporal_p8'):
   values=object_means(pixel,'J_and_F',condition,kind);d[condition]=float(np.mean(list(values.values()))) if values else None
  for name,a,b,source in [('dense_temporal_minus_static','temporal','static',dense_expr),('dense_temporal_minus_shuffled','temporal','shuffled',dense_expr),('final_temporal8_minus_static8','temporal_p8','static_p8',pixel),('temporal8_minus_temporal1','temporal_p8','temporal_p1',pixel)]:
   values=delta(source,'J_and_F',a,b,kind);d[name]=float(np.mean(list(values.values()))) if values else None;d[name+'_ci95']=bootstrap(values) if values else None;d[name+'_objects']=len(values)
  result['by_type'][kind]=d
 dynamic=result['by_type']['dynamic'];pilot_go=all(dynamic[x] is not None and dynamic[x]>0 for x in ('dense_temporal_minus_static','dense_temporal_minus_shuffled','final_temporal8_minus_static8'))
 if not coverage['complete']:decision='INCOMPLETE'
 elif args.phase=='pilot':decision='GO_TO_CONFIRM210' if pilot_go else 'NO-GO'
 else:
  ci=dynamic['final_temporal8_minus_static8_ci95'];dense_ci=dynamic['dense_temporal_minus_static_ci95'];shuffle_ci=dynamic['dense_temporal_minus_shuffled_ci95']
  decision='GO' if dynamic['final_temporal8_minus_static8']>0 and ci[0]>0 and dense_ci[0]>0 and shuffle_ci[0]>0 else ('MIXED' if dense_ci[0]>0 and dynamic['final_temporal8_minus_static8']<=0 else 'NO-GO')
 result['decision']=decision
 losses=[row for row in pixel if row['condition']=='temporal_p8' and row['description_type']=='dynamic'];result['pixel_losses']={'dense_to_init':float(np.mean([x['dense_to_init_loss_J_and_F'] for x in losses])) if losses else None,'init_to_final':float(np.mean([x['init_to_final_loss_J_and_F'] for x in losses])) if losses else None,'latency_seconds':float(np.mean([x['latency_seconds_synchronized'] for x in losses])) if losses else None,'peak_memory_bytes':max([x['peak_memory_bytes'] for x in losses],default=None)}
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with (output/'paired_dynamic_results.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=['video_id','object_id','temporal8_minus_static8']);w.writeheader();[w.writerow({'video_id':k[0],'object_id':k[1],'temporal8_minus_static8':v}) for k,v in delta(pixel,'J_and_F','temporal_p8','static_p8','dynamic').items()]
 print(json.dumps(result,indent=2));return 0


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--dense-root',required=True);p.add_argument('--pixel-root',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=('pilot','confirm'),required=True);return p.parse_args()
if __name__=='__main__':raise SystemExit(run(parse_args()))

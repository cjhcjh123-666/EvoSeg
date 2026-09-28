"""Persistent smoke -> train -> pilot -> gated confirm TDSP queue."""
from __future__ import annotations
import argparse,json,os,subprocess,time,traceback
from datetime import datetime
from pathlib import Path

PY_SA='/tmp/EvoSeg-temporal-process-20260921/.venv/bin/python'
PY_SAM='/9950backfile/chenjiahui/evo_artifacts/envs/sam31/bin/python'
REPO='/tmp/EvoSeg-temporal-dense-prompt'
SAM_REPO='/9950backfile/chenjiahui/evo_artifacts/external/sam3'
CKPT='/9950backfile/chenjiahui/evo_artifacts/models/SAM3.1-official-mirror/sam3.1_multiplex.pt'
SHA='0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
TRAIN='/9950backfile/chenjiahui/evo_artifacts/results/temporal_grounding_mechanism/20260928_mechanism_pilot64/train_manifest.json'
VAL='/9950backfile/chenjiahui/evo_artifacts/results/temporal_seg/20260921_2345_sa2va4b_temporal_process/manifest.json'
TRAIN_TEMP='/9950backfile/chenjiahui/evo_artifacts/results/temporal_grounding_mechanism/20260928_mechanism_pilot64/train_states/stage_grounding_records.jsonl'
VAL_TEMP='/9950backfile/chenjiahui/evo_artifacts/results/temporal_grounding_mechanism_v2/20260928_full_probe_pixel_execution/full_states_merged/stage_grounding_records.jsonl'


class Queue:
 def __init__(self,root):self.root=Path(root);self.logs=self.root/'logs';self.logs.mkdir(parents=True,exist_ok=True);self.started=time.time();self.phase='waiting_static_states';self.processes=[];self.last_progress=0
 def write(self,state='running',extra=None):
  now=time.time();active=[{'pid':p.pid,'name':name,'returncode':p.poll()} for name,p,_ in self.processes if p.poll() is None]
  value={'state':state,'phase':self.phase,'pid':os.getpid(),'tmux_session':'tdsp_20260929_004110','started_at':datetime.fromtimestamp(self.started).astimezone().isoformat(),'updated_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':now-self.started,'active_processes':active};value.update(extra or {})
  temporary=self.root/'STATUS.json.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(self.root/'STATUS.json')
  if now-self.last_progress>=7200 or not (self.root/'PROGRESS.md').exists():
   (self.root/'PROGRESS.md').write_text(f"# TDSP progress\n\n- Updated: {value['updated_at']}\n- Phase: `{self.phase}`\n- State: `{state}`\n- Active: {len(active)}\n- Elapsed: {(now-self.started)/3600:.2f} h\n");self.last_progress=now
  local=datetime.now().astimezone()
  if local.hour>=8 and not (self.root/'MORNING_SNAPSHOT.md').exists() and state=='running':
   (self.root/'MORNING_SNAPSHOT.md').write_text(f"# Morning snapshot\n\n- Time: {value['updated_at']}\n- Phase: `{self.phase}`\n- Active workers: {len(active)}\n- Elapsed: {(now-self.started)/3600:.2f} h\n- Legal background work was left running. See `STATUS.json` for live completion and ETA.\n")
 def spawn(self,name,command,gpu,python=PY_SA):
  log=(self.logs/f'{name}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1');p=subprocess.Popen([python,'-m',*command],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT);self.processes.append((name,p,log));return p
 def wait(self,items):
  while any(p.poll() is None for p in items):self.write();time.sleep(30)
  bad=[p.returncode for p in items if p.returncode]
  for _,p,h in self.processes:
   if p in items:h.close()
  if bad:raise RuntimeError(f'worker failures: {bad}')
 def group(self,specs):
  items=[self.spawn(*spec) for spec in specs];self.wait(items)


def feature_command(manifest,output,shard):
 return ['projects.evoseg.temporal_dense_prompt.extract_sam_features','--manifest',manifest,'--output',str(output),'--sam3-repo',SAM_REPO,'--checkpoint',CKPT,'--expected-checkpoint-sha256',SHA,'--device','0','--shard-index',str(shard),'--num-shards','4']


def dense_command(root,manifest,features,output,start,stop,shard):
 checkpoints=[str(root/f'training/seed{x}/checkpoint.pt') for x in (11,23,42)]
 return ['projects.evoseg.temporal_dense_prompt.evaluate_dense','--manifest',manifest,'--temporal-records',VAL_TEMP,'--static-records',str(root/'val_static_states'),'--features',str(features),'--checkpoints',*checkpoints,'--output',str(output),'--device','0','--object-start',str(start),'--object-stop',str(stop),'--shard-index',str(shard),'--num-shards','4']


def pixel_command(manifest,dense,output,start,stop,shard,shards=4):
 return ['projects.evoseg.temporal_dense_prompt.run_pixel','--manifest',manifest,'--dense-root',str(dense),'--output',str(output),'--sam3-repo',SAM_REPO,'--checkpoint',CKPT,'--expected-checkpoint-sha256',SHA,'--device','0','--object-start',str(start),'--object-stop',str(stop),'--shard-index',str(shard),'--num-shards',str(shards)]


def run(args):
 q=Queue(args.run_dir);root=q.root
 try:
  while True:
   statuses=list((root/'val_static_states').glob('v12shard*/STAGE_STATUS.json'))
   if len(statuses)==12 and all(json.loads(x.read_text()).get('state','').startswith('complete') for x in statuses):break
   q.write(extra={'static_validation_shards_complete':sum(json.loads(x.read_text()).get('state','').startswith('complete') for x in statuses),'static_validation_shards_planned':12});time.sleep(30)
  q.phase='extract_train_sam31_features';q.group([(f'train_features_{i}',feature_command(TRAIN,root/'train_features',i),i+2,PY_SAM) for i in range(4)])
  q.phase='extract_validation_sam31_features';q.group([(f'val_features_{i}',feature_command(VAL,root/'val_features',i),i+2,PY_SAM) for i in range(4)])
  q.phase='two_video_training_smoke';q.group([('smoke_train',['projects.evoseg.temporal_dense_prompt.smoke_train','--manifest',TRAIN,'--temporal-records',TRAIN_TEMP,'--static-records',str(root/'train_static_states'),'--features',str(root/'train_features'),'--output',str(root/'smoke/training_smoke.json'),'--device','0'],2,PY_SA)])
  q.phase='train_three_seeds';spec=[]
  for seed,gpu in zip((11,23,42),(2,3,4)):
   command=['projects.evoseg.temporal_dense_prompt.train','--manifest',TRAIN,'--temporal-records',TRAIN_TEMP,'--static-records',str(root/'train_static_states'),'--features',str(root/'train_features'),'--output',str(root/f'training/seed{seed}'),'--seed',str(seed),'--device','0']
   spec.append((f'train_seed{seed}',command,gpu,PY_SA))
  q.group(spec)
  q.phase='pilot64_dense';q.group([(f'pilot_dense_{i}',dense_command(root,VAL,root/'val_features',root/'pilot64/dense',0,64,i),i+2,PY_SA) for i in range(4)])
  q.phase='two_video_public_point_smoke';q.group([('smoke_pixel',pixel_command(VAL,root/'pilot64/dense',root/'smoke/pixel',0,2,0,1),2,PY_SAM)])
  smoke_rows=[json.loads(line) for line in (root/'smoke/pixel/pixel_results.shard0.jsonl').open() if line.strip() and json.loads(line).get('status')=='success']
  single_peak=max(row['peak_memory_bytes'] for row in smoke_rows);pixel_workers=8 if single_peak < 30*2**30 else 4
  q.write(extra={'sam31_single_worker_peak_bytes':single_peak,'sam31_workers_per_gpu':pixel_workers//4})
  q.phase='pilot64_pixel';q.group([(f'pilot_pixel_{i}',pixel_command(VAL,root/'pilot64/dense',root/'pilot64/pixel',0,64,i,pixel_workers),2+i%4,PY_SAM) for i in range(pixel_workers)])
  q.phase='pilot64_summary';subprocess.run([PY_SA,'-m','projects.evoseg.temporal_dense_prompt.summarize','--dense-root',str(root/'pilot64/dense'),'--pixel-root',str(root/'pilot64/pixel'),'--output',str(root/'pilot64/summary'),'--phase','pilot'],cwd=REPO,check=True)
  pilot=json.loads((root/'pilot64/summary/summary.json').read_text())
  if pilot['decision']!='GO_TO_CONFIRM210':
   q.phase='finalize_reports';subprocess.run([PY_SA,'-m','projects.evoseg.temporal_dense_prompt.finalize','--run-dir',str(root),'--docs',str(Path(REPO)/'docs/temporal_dense_prompt'),'--manifest',VAL],cwd=REPO,check=True);q.write('complete',{'decision':'NO-GO','pilot':pilot});return 0
  q.phase='confirm210_dense';q.group([(f'confirm_dense_{i}',dense_command(root,VAL,root/'val_features',root/'confirm210/dense',64,274,i),i+2,PY_SA) for i in range(4)])
  q.phase='confirm210_pixel';q.group([(f'confirm_pixel_{i}',pixel_command(VAL,root/'confirm210/dense',root/'confirm210/pixel',64,274,i,pixel_workers),2+i%4,PY_SAM) for i in range(pixel_workers)])
  q.phase='confirm210_summary';subprocess.run([PY_SA,'-m','projects.evoseg.temporal_dense_prompt.summarize','--dense-root',str(root/'confirm210/dense'),'--pixel-root',str(root/'confirm210/pixel'),'--output',str(root/'confirm210/summary'),'--phase','confirm'],cwd=REPO,check=True)
  final=json.loads((root/'confirm210/summary/summary.json').read_text());q.phase='finalize_reports';subprocess.run([PY_SA,'-m','projects.evoseg.temporal_dense_prompt.finalize','--run-dir',str(root),'--docs',str(Path(REPO)/'docs/temporal_dense_prompt'),'--manifest',VAL],cwd=REPO,check=True);q.write('complete',{'decision':final['decision'],'pilot':pilot,'confirm':final});return 0
 except Exception as error:
  q.write('failed',{'error':str(error),'traceback':traceback.format_exc()});raise


def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);return p.parse_args()
if __name__=='__main__':raise SystemExit(run(parse_args()))

"""Run and report the separate GT-free GroundMoRe concept candidate audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temporary.replace(path)


def update(root: Path, phase: str, **values):
    atomic(root/"STATUS.json",{"state":"running","phase":phase,"pid":os.getpid(),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**values})


def wait(processes,root,phase):
    while any(value.poll() is None for value in processes):
        update(root,phase,workers=len(processes),running=sum(value.poll() is None for value in processes));time.sleep(15)
    return [value.returncode for value in processes]


def rows(path: Path):
    with path.open() as handle:return list(csv.DictReader(handle))


def metrics(values,total):
    jf=[float(row["oracle_J_and_F"]) for row in values]
    return {"total_official_expressions":total,"evaluated":len(values),"unresolved_or_failed":total-len(values),
            "nonempty_candidates":sum(int(row["candidate_count"])>0 for row in values),
            "recall_at_0_3":sum(value>=.3 for value in jf)/total,"recall_at_0_5":sum(value>=.5 for value in jf)/total,
            "recall_at_0_7":sum(value>=.7 for value in jf)/total,"oracle_jf_all":sum(jf)/total}


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);logs=root/"logs";logs.mkdir(exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text());total=sum(len(item["expressions"]) for item in manifest["objects"])
    processes=[]
    for shard in range(args.workers):
        destination=root/f"shard-{shard}";destination.mkdir(exist_ok=True);device=args.device_offset+shard
        command=[args.sam_python,"-m","projects.evoseg.temporal_compiler.sam31_candidate_protocol","generate","--manifest",args.manifest,"--run-dir",str(destination),"--checkpoint",args.checkpoint,"--expected-checkpoint-sha256","0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6","--checkpoint-source","AEmotionStudio/sam3.1-byte-identical-official-sha","--sam3-repo",args.sam_repo,"--device",str(device),"--max-objects",str(len(manifest["objects"])),"--shard-index",str(shard),"--num-shards",str(args.workers),"--max-num-objects","16","--multiplex-count","16","--prompt-methods","groundmore_concept","--no-use-fa3","--no-compile"]
        handle=(logs/f"worker_{shard}.log").open("a");processes.append(subprocess.Popen(command,cwd=args.repo,stdout=handle,stderr=subprocess.STDOUT))
    codes=wait(processes,root,"concept_candidate_generation")
    evaluated=root/"evaluated";additional=[]
    for shard in range(1,args.workers):additional.extend(["--additional-run-dir",str(root/f"shard-{shard}")])
    command=[args.sam_python,"-m","projects.evoseg.temporal_compiler.sam31_candidate_protocol","evaluate","--manifest",args.manifest,"--run-dir",str(root/"shard-0"),*additional,"--output-dir",str(evaluated),"--workers","32"]
    update(root,"concept_candidate_evaluation",generation_return_codes=codes)
    with (logs/"evaluate.log").open("a") as handle:subprocess.run(command,cwd=args.repo,stdout=handle,stderr=subprocess.STDOUT,check=True)
    concept=rows(evaluated/"sam31_candidate_metrics.csv");raw=rows(Path(args.raw_metrics))
    raw=[row for row in raw if row["prompt_method"]=="raw_expression" and row.get("dataset")=="groundmore"]
    summary={"raw_expression":metrics(raw,total),"deterministic_concept":metrics(concept,total),
             "generation_return_codes":codes,"ground_truth_used_for_prompt":False,"prompt_rule":"who/whom/whose => person; explicit subject noun phrase => parsed noun; unresolved => no prompt"}
    atomic(root/"candidate_generation_audit.json",summary)
    fields=["condition",*summary["raw_expression"].keys()]
    with (root/"candidate_generation_audit.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for condition in ("raw_expression","deterministic_concept"):writer.writerow({"condition":condition,**summary[condition]})
    raw_m=summary["raw_expression"];concept_m=summary["deterministic_concept"]
    (root/"CANDIDATE_GENERATION_AUDIT.md").write_text(f"""# GroundMoRe candidate-generation improvement audit

This is an inference-preprocessing audit, not an OPG-v2 contribution. The deterministic parser reads official question text only. It maps person interrogatives to broad `person`, uses an explicit parsed noun phrase when available, and leaves unresolved questions without a prompt. GT object IDs, answers, and masks are never read during prompt construction.

| condition | evaluated / 480 | nonempty | Recall@0.3 | Recall@0.5 | Recall@0.7 | oracle J&F (all 480) |
|---|---:|---:|---:|---:|---:|---:|
| Raw full expression | {raw_m['evaluated']} / {total} | {raw_m['nonempty_candidates']} | {raw_m['recall_at_0_3']*100:.2f}% | {raw_m['recall_at_0_5']*100:.2f}% | {raw_m['recall_at_0_7']*100:.2f}% | {raw_m['oracle_jf_all']*100:.2f} |
| Deterministic concept | {concept_m['evaluated']} / {total} | {concept_m['nonempty_candidates']} | {concept_m['recall_at_0_3']*100:.2f}% | {concept_m['recall_at_0_5']*100:.2f}% | {concept_m['recall_at_0_7']*100:.2f}% | {concept_m['oracle_jf_all']*100:.2f} |

Unresolved or failed deterministic prompts: {concept_m['unresolved_or_failed']}. Candidate-generation changes are reported separately and are not attributed to ordered reasoning.
""")
    atomic(root/"STATUS.json",{"state":"complete","phase":"complete","pid":os.getpid(),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"summary":summary})


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","output","repo","sam_python","checkpoint","sam_repo","raw_metrics"):p.add_argument(f"--{name.replace('_','-')}",required=True)
    p.add_argument("--workers",type=int,default=7);p.add_argument("--device-offset",type=int,default=1);return p.parse_args()


if __name__=="__main__":run(parse_args())

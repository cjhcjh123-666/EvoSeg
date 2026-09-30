"""Persistent, resumable controller for the preregistered OPG SFT run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


EXPECTED_ARCHIVE_BYTES=30219642880
EXPECTED_ARCHIVE_SHA="504fe81fe822f978f1956114d241ce317886f8928eca028d3cc2e2aa0ab2ce55"


def atomic(path:Path,value:dict):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temp.replace(path)


def update(root:Path,phase:str,**extra):
    value={"state":"running","phase":phase,"pid":os.getpid(),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**extra};atomic(root/"STATUS.json",value)
    (root/"PROGRESS.md").write_text(f"# OPG SFT progress\n\n- Updated: {value['updated_at']}\n- Phase: `{phase}`\n"+"".join(f"- {key}: {item}\n" for key,item in extra.items()))


def run_logged(command:list[str],log:Path,cwd:Path,root:Path,phase:str):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as handle:
        process=subprocess.Popen(command,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,text=True)
        while process.poll() is None:
            update(root,phase,child_pid=process.pid,command=command);time.sleep(30)
    if process.returncode:
        raise RuntimeError(f"phase {phase} failed ({process.returncode}); see {log}")


def wait_many(processes,root:Path,phase:str):
    while any(process.poll() is None for process in processes):
        update(root,phase,workers=len(processes),running=sum(process.poll() is None for process in processes));time.sleep(30)
    return [process.returncode for process in processes]


def file_sha(path:Path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(16*1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def combine_manifests(train_path:Path,test_path:Path,output:Path):
    train=json.loads(train_path.read_text());test=json.loads(test_path.read_text())
    value={"dataset":{**train["dataset"],"split":"trainval+test"},"selection":{"q_type":"official Sequential only","manual_or_llm_classification":False},"objects":train["objects"]+test["objects"]}
    output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");return value


def run(args):
    root=Path(args.artifact).resolve();root.mkdir(parents=True,exist_ok=True);repo=Path(args.repo).resolve();data=Path(args.data_root).resolve()
    atomic(root/"run_config.json",{
        "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"command":[os.sys.executable,*os.sys.argv],
        "branch":"research/ordered-process-grounding-sft","code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        "seeds":[11,23,42],"process_slots":4,"track_steps":8,"margin":0.2,"lambda_order":1.0,
        "frozen_components":["Sa2VA","SAM3.1","candidate generator","visual feature encoder"],
        "groundmore_official_repo_commit":"d5074ab920a86a7ea92ecc69902109f6887f3e10","arguments":vars(args),
    })
    archive=data/"groundmore_v2.tar";control=Path(str(archive)+".aria2");logs=root/"logs";logs.mkdir(exist_ok=True)
    while control.exists() or not archive.exists() or archive.stat().st_size!=EXPECTED_ARCHIVE_BYTES:
        update(root,"wait_groundmore_download",allocated_bytes=archive.stat().st_size if archive.exists() else 0,expected_bytes=EXPECTED_ARCHIVE_BYTES);time.sleep(30)
    update(root,"verify_groundmore_archive")
    actual=file_sha(archive)
    if actual!=EXPECTED_ARCHIVE_SHA:raise RuntimeError(f"GroundMoRe archive SHA mismatch: {actual}")
    extraction_marker=data/f".groundmore_v2_extracted_{EXPECTED_ARCHIVE_SHA}.json"
    if not extraction_marker.is_file():
        run_logged(["tar","-xf",str(archive),"-C",str(data)],logs/"groundmore_extract.log",repo,root,"extract_groundmore")
        atomic(extraction_marker,{
            "archive":str(archive),"sha256":actual,"bytes":archive.stat().st_size,
            "completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    train_manifest=root/"groundmore_trainval_sequential_manifest.json";test_manifest=root/"groundmore_test_sequential_manifest.json"
    python=args.python
    common=[python,"-m","projects.evoseg.ordered_process_grounding.build_groundmore_manifest"]
    if not train_manifest.exists():
        run_logged(common+["--metadata",str(data/"trainval_v2.json"),"--data-root",str(data),"--binary-mask-root",str(root/"groundmore_binary_masks"),"--split","trainval","--output",str(train_manifest)],logs/"manifest_train.log",repo,root,"build_groundmore_train_manifest")
    if not test_manifest.exists():
        run_logged(common+["--metadata",str(data/"test_v2.json"),"--data-root",str(data),"--binary-mask-root",str(root/"groundmore_binary_masks"),"--split","test","--output",str(test_manifest)],logs/"manifest_test.log",repo,root,"build_groundmore_test_manifest")
    combined_path=root/"groundmore_sequential_manifest.json";combined=combine_manifests(train_manifest,test_manifest,combined_path);objects=len(combined["objects"])

    candidate_root=root/"groundmore_candidates";candidate_root.mkdir(exist_ok=True);workers=[]
    for shard in range(args.candidate_workers):
        shard_root=candidate_root/f"shard-{shard}";shard_root.mkdir(exist_ok=True)
        command=[args.sam_python,"-m","projects.evoseg.temporal_compiler.sam31_candidate_protocol","generate","--manifest",str(combined_path),"--run-dir",str(shard_root),"--checkpoint",args.sam_checkpoint,"--expected-checkpoint-sha256","0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6","--checkpoint-source","AEmotionStudio/sam3.1-byte-identical-official-sha","--sam3-repo",args.sam_repo,"--device",str(shard),"--max-objects",str(objects),"--shard-index",str(shard),"--num-shards",str(args.candidate_workers),"--max-num-objects","16","--multiplex-count","16","--prompt-methods","raw_expression","--no-use-fa3","--no-compile"]
        handle=(logs/f"ground_candidates_{shard}.log").open("a");workers.append(subprocess.Popen(command,cwd=repo,stdout=handle,stderr=subprocess.STDOUT))
    codes=wait_many(workers,root,"groundmore_candidate_generation")
    # One deterministic retry retains every failed attempt and resumes successes.
    if any(codes):
        time.sleep(60);workers=[]
        for shard in range(args.candidate_workers):
            status=candidate_root/f"shard-{shard}"/"STATUS.json"
            if status.exists() and json.loads(status.read_text()).get("state")=="complete":continue
            command=json.loads((candidate_root/f"shard-{shard}"/"run_config.json").read_text())["command"]
            handle=(logs/f"ground_candidates_{shard}_retry.log").open("a");workers.append(subprocess.Popen(command,cwd=repo,stdout=handle,stderr=subprocess.STDOUT))
        if workers:wait_many(workers,root,"groundmore_candidate_retry")

    evaluated=candidate_root/"evaluated";additional=[]
    for shard in range(1,args.candidate_workers):additional.extend(["--additional-run-dir",str(candidate_root/f"shard-{shard}")])
    run_logged([args.sam_python,"-m","projects.evoseg.temporal_compiler.sam31_candidate_protocol","evaluate","--manifest",str(combined_path),"--run-dir",str(candidate_root/"shard-0"),*additional,"--output-dir",str(evaluated),"--workers","32"],logs/"ground_candidates_eval.log",repo,root,"evaluate_groundmore_candidates")

    feature_parts=[];workers=[]
    for shard in range(args.feature_workers):
        output=root/f"ground_features_shard{shard}.pt";feature_parts.append(output)
        command=[python,"-m","projects.evoseg.ordered_process_grounding.prepare_groundmore_features","--manifest",str(combined_path),"--candidate-root",str(candidate_root),"--prompt-method","raw_expression","--sa2va-model",args.sa2va_model,"--vision-checkpoint",args.vision_checkpoint,"--output",str(output),"--device",str(shard),"--track-steps","8","--image-batch-size","32","--shard-index",str(shard),"--num-shards",str(args.feature_workers)]
        handle=(logs/f"ground_features_{shard}.log").open("a");workers.append(subprocess.Popen(command,cwd=repo,stdout=handle,stderr=subprocess.STDOUT))
    codes=wait_many(workers,root,"groundmore_feature_extraction")
    if any(codes):raise RuntimeError(f"GroundMoRe feature workers failed: {codes}")
    merged=root/"ground_features.pt"
    run_logged([python,"-m","projects.evoseg.ordered_process_grounding.merge_features","--input",*[str(x) for x in feature_parts],"--output",str(merged)],logs/"ground_features_merge.log",repo,root,"merge_groundmore_features")
    ground_train=root/"ground_train_features.pt";ground_test=root/"ground_test_features.pt"
    run_logged([python,"-m","projects.evoseg.ordered_process_grounding.split_features","--input",str(merged),"--train-output",str(ground_train),"--test-output",str(ground_test)],logs/"ground_features_split.log",repo,root,"split_groundmore_features")

    command=[python,"-m","projects.evoseg.ordered_process_grounding.train_sft","--train-features",args.long_train_features,"--train-metrics",args.long_train_metrics,"--eval-features",args.long_eval_features,"--eval-metrics",args.long_eval_metrics,"--mean-checkpoints",args.mean_checkpoints,"--prior-ftsg-results",args.prior_ftsg_results,"--ground-train-features",str(ground_train),"--ground-train-metrics",str(evaluated/"sam31_candidate_metrics.csv"),"--ground-eval-features",str(ground_test),"--ground-eval-metrics",str(evaluated/"sam31_candidate_metrics.csv"),"--output",str(root),"--device","cuda:0"]
    run_logged(command,logs/"joint_sft.log",repo,root,"joint_sft")
    docs=repo/"docs"/"ordered_process_grounding"
    run_logged([python,"-m","projects.evoseg.ordered_process_grounding.finalize","--artifact",str(root),"--docs",str(docs),"--long-manifest",args.long_manifest],logs/"finalize.log",repo,root,"finalize")
    run_logged(["git","diff","--check"],logs/"git_diff_check.log",repo,root,"git_diff_check")
    run_logged([python,"-m","pytest","-q","--import-mode=importlib","projects/evoseg/ordered_process_grounding/tests","projects/evoseg/temporal_compiler/tests"],logs/"pytest.log",repo,root,"pytest")
    subprocess.run(["git","add","docs/ordered_process_grounding"],cwd=repo,check=True)
    subprocess.run(["git","commit","-m","add ordered process grounding SFT results"],cwd=repo,check=True)
    subprocess.run(["git","push","origin-ssh","research/ordered-process-grounding-sft"],cwd=repo,check=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip();decision=json.loads((root/"FINAL_DECISION.json").read_text())["decision"]
    atomic(root/"STATUS.json",{"state":"complete","phase":"complete","pid":os.getpid(),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"decision":decision,"commit":commit})


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("artifact","repo","data_root","python","sam_python","sam_checkpoint","sam_repo","sa2va_model","vision_checkpoint","long_train_features","long_train_metrics","long_eval_features","long_eval_metrics","mean_checkpoints","prior_ftsg_results","long_manifest"):p.add_argument(f"--{name.replace('_','-')}",required=True)
    p.add_argument("--candidate-workers",type=int,default=8);p.add_argument("--feature-workers",type=int,default=4);return p.parse_args()


if __name__=="__main__":run(parse_args())

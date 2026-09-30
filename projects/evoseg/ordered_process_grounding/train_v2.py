"""Train and evaluate Discriminative Ordered Process Grounding v2."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.factorized_temporal_spatial.protocol import load_mean_pool
from projects.evoseg.ordered_process_grounding.model import count_trainable_parameters
from projects.evoseg.ordered_process_grounding.model_v2 import DiscriminativeOrderedHead
from projects.evoseg.ordered_process_grounding.protocol import block_swap_order, fixed_order_negative, is_order_sensitive, reverse_order
from projects.evoseg.ordered_process_grounding.train_sft import (
    base_logits,
    load_ground_base,
    opg_examples,
    order_inputs,
    order_supervised,
    split_train,
)
from projects.evoseg.temporal_grounding_mechanism.common import object_weighted, source_video_bootstrap, write_csv


VARIANTS = {
    "v2_no_relative": {"alignment": "monotonic", "candidate": 1.0, "relative": 0.0, "self": 1.0, "delta_fusion": True},
    "v2_no_candidate": {"alignment": "monotonic", "candidate": 0.0, "relative": 1.0, "self": 1.0, "delta_fusion": True},
    "v2_no_monotonic": {"alignment": "global", "candidate": 1.0, "relative": 1.0, "self": 1.0, "delta_fusion": True},
    "v2_no_delta_fusion": {"alignment": "monotonic", "candidate": 1.0, "relative": 1.0, "self": 1.0, "delta_fusion": False},
    "opg_v2_full": {"alignment": "monotonic", "candidate": 1.0, "relative": 1.0, "self": 1.0, "delta_fusion": True},
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def update_status(root: Path, phase: str, **values) -> None:
    atomic_json(root / "STATUS.json", {
        "state": "running", "phase": phase, "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values,
    })


def permutation_tracks(row: dict, device: str, kind: str | None = None):
    if kind == "reverse":
        permutation = reverse_order(8)
    elif kind == "block_swap":
        permutation = block_swap_order(8)
    else:
        _, permutation = fixed_order_negative(row["identity"], 8)
    return order_inputs(row, device, permutation)[1]


def compute(model, base_model, row: dict, device: str, permutation: str | None = None):
    base = base_logits(base_model, row, device).unsqueeze(0)
    query, tracks = order_inputs(row, device)
    permuted = permutation_tracks(row, device, permutation)
    return model(base, query, tracks, permuted)


def losses(model, base_model, row: dict, device: str, config: dict, margin: float, warmup: bool = False):
    import torch
    import torch.nn.functional as functional

    final, audit = compute(model, base_model, row, device)
    target = torch.tensor([row["target_index"]], device=device)
    identity = functional.cross_entropy(final, target)
    candidate = functional.cross_entropy(audit["order"], target)
    relative = torch.zeros((), device=device)
    self_order = torch.zeros((), device=device)
    if order_supervised(row):
        relative = functional.cross_entropy(audit["delta"], target)
        index = row["target_index"]
        self_order = functional.relu(margin - audit["order"][0, index] + audit["order_permuted"][0, index])
    if warmup:
        total = identity
    else:
        total = identity + config["candidate"] * candidate
        if order_supervised(row):
            total = total + config["relative"] * relative + config["self"] * self_order
    return total, {
        "identity": float(identity.detach().cpu()),
        "candidate": float(candidate.detach().cpu()),
        "relative": float(relative.detach().cpu()),
        "self_order": float(self_order.detach().cpu()),
        "beta_order": float(audit["beta_order"].detach().cpu()),
        "beta_delta": float(audit["beta_delta"].detach().cpu()),
        "gate_mean": float(audit["gates"].mean().detach().cpu()),
    }


def validation_loss(model, base_model, rows, device, config, margin, warmup=False):
    import torch
    model.eval(); values=[]; parts=defaultdict(list)
    with torch.inference_mode():
        for row in rows:
            total, audit = losses(model, base_model, row, device, config, margin, warmup)
            values.append(float(total.cpu()))
            for key, value in audit.items(): parts[key].append(value)
    return float(np.mean(values)), {key: float(np.mean(value)) for key, value in parts.items()}


def train_one(fitting, validation, base_model, seed: int, variant: str, args):
    import torch
    config = VARIANTS[variant]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = DiscriminativeOrderedHead(alignment=config["alignment"], use_delta_fusion=config["delta_fusion"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator=random.Random(seed); history=[]; started=time.monotonic()
    if str(args.device).startswith("cuda"): torch.cuda.reset_peak_memory_stats(args.device)
    for stage, epochs, warmup in (("identity_warmup", 1, True), ("ordered_sft", args.epochs, False)):
        best=None; best_loss=float("inf"); stale=0
        for epoch in range(epochs):
            model.train(); indices=list(range(len(fitting))); generator.shuffle(indices); records=[]
            for index in indices:
                total, audit = losses(model, base_model, fitting[index], args.device, config, args.margin, warmup)
                optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                records.append({"total":float(total.detach().cpu()),**audit})
            valid, valid_parts=validation_loss(model,base_model,validation,args.device,config,args.margin,warmup)
            record={"stage":stage,"epoch":epoch+1,"train_total":float(np.mean([x["total"] for x in records])),"validation_total":valid,
                    "beta_order":float(model.beta_order.detach().cpu()),"beta_delta":float(model.beta_delta.detach().cpu()),
                    **{f"validation_{key}":value for key,value in valid_parts.items()}}
            history.append(record)
            if warmup:
                best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}; continue
            if valid < best_loss - 1e-5:
                best_loss=valid; best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}; stale=0
            else:
                stale+=1
                if stale>=args.patience: break
        model.load_state_dict(best)
    return model.eval(), {
        "seed":seed,"variant":variant,"history":history,"elapsed_seconds":time.monotonic()-started,
        "trainable_parameters":count_trainable_parameters(model),
        "peak_memory_bytes":int(torch.cuda.max_memory_allocated(args.device)) if str(args.device).startswith("cuda") else 0,
        "final_beta_order":float(model.beta_order.detach().cpu()),"final_beta_delta":float(model.beta_delta.detach().cpu()),
    }


def save_model(model, audit, path: Path):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"seed":audit["seed"],"variant":audit["variant"],
                "trainable_parameters":audit["trainable_parameters"],"final_beta_order":audit["final_beta_order"],
                "final_beta_delta":audit["final_beta_delta"]},path)
    path.with_suffix(".json").write_text(json.dumps(audit,indent=2)+"\n")


def load_model(path: Path, device: str):
    import torch
    payload=torch.load(path,map_location="cpu"); config=VARIANTS[payload["variant"]]
    model=DiscriminativeOrderedHead(alignment=config["alignment"],use_delta_fusion=config["delta_fusion"]).to(device)
    model.load_state_dict(payload["state_dict"]); return model.eval(),payload


def ensemble_logits(models, base_models, row, device):
    values=defaultdict(list)
    import torch
    with torch.inference_mode():
        for model,base in zip(models,base_models):
            final,audit=compute(model,base,row,device)
            values["fused"].append(final[0].float().cpu().numpy())
            values["order"].append(audit["order"][0].float().cpu().numpy())
            values["delta"].append(audit["delta"][0].float().cpu().numpy())
            values["base"].append(audit["base"][0].float().cpu().numpy())
    return {key:np.mean(value,axis=0) for key,value in values.items()}


def selected_row(row, method: str, logits: np.ndarray):
    ids=[int(value) for value in row.get("candidate_track_ids",[])]; index=int(np.argmax(logits)) if len(logits) else None
    selected=ids[index] if index is not None else None
    return {
        "identity":row["identity"],"dataset":row["dataset"],"split":row["split"],"video_id":row["video_id"],
        "object_id":row["object_id"],"expression_id":row["expression_id"],"description_type":row["description_type"],
        "expression":row["expression"],"explicit_order":int(order_supervised(row)),"method":method,
        "candidate_hit":int(row["candidate_hit"]),"candidate_count":len(ids),"oracle_track_id":row["oracle_track_id"],
        "selected_track_id":selected,"selection_correct":int(row["candidate_hit"] and selected==row["oracle_track_id"]),
        "J":float(row["candidate_j"].get(selected,0.0)),"F":float(row["candidate_f"].get(selected,0.0)),
        "J_and_F":float(row["candidate_jf"].get(selected,0.0)),
    }


def add_prior_v1(rows: list[dict], path: Path, dataset: str):
    if not path.is_file(): return
    with path.open() as handle:
        for value in csv.DictReader(handle):
            if value["method"]!="opg_full" or value.get("seed")!="ensemble": continue
            if value.get("dataset")!=dataset: continue
            copy=dict(value); copy["method"]="opg_v1"; copy["explicit_order"]=int(is_order_sensitive(value["expression"]) or dataset=="groundmore")
            for key in ("candidate_hit","candidate_count","selection_correct"): copy[key]=int(float(copy[key]))
            for key in ("J","F","J_and_F"): copy[key]=float(copy[key])
            rows.append(copy)


def summarize(rows, kind: str, hit_only: bool=False):
    subset=[row for row in rows if (kind=="explicit_order" and row["explicit_order"] or row["description_type"]==kind)]
    if hit_only: subset=[row for row in subset if row["candidate_hit"]]
    output=[]
    for method in sorted({row["method"] for row in subset}):
        chosen=[row for row in subset if row["method"]==method]
        weighted=object_weighted(chosen,["selection_correct","J","F","J_and_F"])
        output.append({"subset":kind,"hit_only":int(hit_only),"method":method,"expressions":len(chosen),"objects":len(weighted),
                       "selection_accuracy":float(np.mean([x["selection_correct"] for x in weighted])) if weighted else 0.0,
                       "J":float(np.mean([x["J"] for x in weighted])) if weighted else 0.0,
                       "F":float(np.mean([x["F"] for x in weighted])) if weighted else 0.0,
                       "J_and_F":float(np.mean([x["J_and_F"] for x in weighted])) if weighted else 0.0})
    return output


def paired_bootstrap(rows, left, right, kind, field, hit_only=False):
    grouped=defaultdict(dict);meta={}
    for row in rows:
        matches=(row["explicit_order"] if kind=="explicit_order" else row["description_type"]==kind)
        if matches and row["method"] in {left,right} and (not hit_only or row["candidate_hit"]):
            grouped[row["identity"]][row["method"]]=float(row[field]);meta[row["identity"]]=row
    paired=[{"video_id":meta[key]["video_id"],"object_id":meta[key]["object_id"],left:value[left],right:value[right]}
            for key,value in grouped.items() if left in value and right in value]
    return source_video_bootstrap(paired,left,right)


def best_other(vector: np.ndarray, target: int) -> float:
    values=np.delete(vector,target)
    return float(values.max()) if len(values) else float(vector[target])


def order_diagnostics(models,base_models,rows,device):
    import torch
    output=[]
    with torch.inference_mode():
        for row in rows:
            if not order_supervised(row) or not row["candidate_hit"] or len(row["candidate_track_ids"])<2: continue
            target=row["target_index"]
            originals=[];reverses=[];blocks=[]
            for model,base in zip(models,base_models):
                originals.append(compute(model,base,row,device)[1]["order"][0].float().cpu().numpy())
                reverses.append(compute(model,base,row,device,"reverse")[1]["order_permuted"][0].float().cpu().numpy())
                blocks.append(compute(model,base,row,device,"block_swap")[1]["order_permuted"][0].float().cpu().numpy())
            original=np.mean(originals,axis=0)
            for name,permuted in (("reverse",np.mean(reverses,axis=0)),("block_swap",np.mean(blocks,axis=0))):
                delta=original-permuted;rank_original=int((-original).argsort().tolist().index(target)+1);rank_permuted=int((-permuted).argsort().tolist().index(target)+1)
                output.append({"identity":row["identity"],"dataset":row["dataset"],"video_id":row["video_id"],"object_id":row["object_id"],
                               "expression_id":row["expression_id"],"permutation":name,
                               "self_order_margin":float(original[target]-permuted[target]),
                               "candidate_margin":float(original[target]-best_other(original,target)),
                               "relative_order_margin":float(delta[target]-best_other(delta,target)),
                               "target_rank_original":rank_original,"target_rank_permuted":rank_permuted,
                               "rank_degradation":rank_permuted-rank_original})
    return output


def diagnostic_summary(rows):
    result={}
    for dataset in sorted({row["dataset"] for row in rows}):
        result[dataset]={}
        for permutation in ("reverse","block_swap"):
            subset=[row for row in rows if row["dataset"]==dataset and row["permutation"]==permutation]
            values={}
            for field in ("self_order_margin","candidate_margin","relative_order_margin","rank_degradation"):
                left=f"{field}_left";right=f"{field}_right"
                pairs=[{"video_id":row["video_id"],left:float(row[field]),right:0.0} for row in subset]
                values[field]=source_video_bootstrap(pairs,left,right)
            result[dataset][permutation]={"samples":len(subset),**values}
    return result


def transition_counts(rows, kind):
    grouped=defaultdict(dict)
    for row in rows:
        matches=(row["explicit_order"] if kind=="explicit_order" else row["description_type"]==kind)
        if matches and row["method"] in {"mean_pool","opg_v2_full"}: grouped[row["identity"]][row["method"]]=row
    fixes=damages=both_correct=both_wrong=0
    for value in grouped.values():
        if len(value)<2: continue
        base=value["mean_pool"]["selection_correct"];full=value["opg_v2_full"]["selection_correct"]
        fixes+=int(full and not base);damages+=int(base and not full);both_correct+=int(base and full);both_wrong+=int(not base and not full)
    return {"paired":len(grouped),"fixes":fixes,"damages":damages,"net_fixes":fixes-damages,"both_correct":both_correct,"both_wrong":both_wrong}


def run(args):
    import torch
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);update_status(root,"load")
    long_train_all,long_train_hits=opg_examples(Path(args.train_features),Path(args.train_metrics),args.hit_threshold,"concept")
    long_eval,_=opg_examples(Path(args.eval_features),Path(args.eval_metrics),args.hit_threshold,"concept")
    ground_train_all,ground_train_hits=opg_examples(Path(args.ground_train_features),Path(args.ground_train_metrics),args.hit_threshold,"raw_expression")
    ground_eval,_=opg_examples(Path(args.ground_eval_features),Path(args.ground_eval_metrics),args.hit_threshold,"raw_expression")
    for row in long_train_all+long_eval: row["_base_kind"]="long_rvos"
    for row in ground_train_all+ground_eval: row["_base_kind"]="groundmore"
    if {x["video_id"] for x in long_train_all}&{x["video_id"] for x in long_eval}: raise RuntimeError("Long-RVOS train/eval overlap")
    if {x["video_id"] for x in ground_train_all}&{x["video_id"] for x in ground_eval}: raise RuntimeError("GroundMoRe train/test overlap")
    long_fit,long_valid=split_train(long_train_hits);ground_fit,ground_valid=split_train(ground_train_hits)
    fitting=long_fit+ground_fit;validation=long_valid+ground_valid
    long_bases=[];ground_bases=[]
    for seed in args.seeds:
        model,payload=load_mean_pool(Path(args.mean_checkpoints)/f"mean_pool_seed{seed}.pt",args.device);model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        long_bases.append(model)
        ground,_=load_ground_base(Path(args.v1_checkpoints)/f"ground_mean_pool_seed{seed}.pt",args.device);ground.eval()
        for parameter in ground.parameters():parameter.requires_grad_(False)
        ground_bases.append(ground)
    base_models=[{"long_rvos":a,"groundmore":b} for a,b in zip(long_bases,ground_bases)]
    model_sets={key:[] for key in VARIANTS};training=[];planned=len(VARIANTS)*len(args.seeds)
    for variant in VARIANTS:
        for seed,base in zip(args.seeds,base_models):
            path=root/"checkpoints"/f"{variant}_seed{seed}.pt"
            if path.is_file(): model,_=load_model(path,args.device);audit=json.loads(path.with_suffix(".json").read_text())
            else: model,audit=train_one(fitting,validation,base,seed,variant,args);save_model(model,audit,path)
            model_sets[variant].append(model);training.append(audit);update_status(root,"training",completed=len(training),planned=planned,variant=variant,seed=seed)
    def evaluate(dataset_rows,dataset):
        update_status(root,f"evaluate_{dataset}",expressions=len(dataset_rows));rows=[];started=time.monotonic()
        for row in dataset_rows:
            full_values=ensemble_logits(model_sets["opg_v2_full"],base_models,row,args.device) if row["tracks"].shape[0] else {x:np.zeros(0) for x in ("base","order","delta","fused")}
            for selector,key in (("mean_pool","base"),("opg_v2_order_only","order"),("opg_v2_delta_only","delta"),("opg_v2_full","fused")):
                rows.append(selected_row(row,selector,full_values[key]))
            for variant in VARIANTS:
                if variant=="opg_v2_full":continue
                values=ensemble_logits(model_sets[variant],base_models,row,args.device) if row["tracks"].shape[0] else {"fused":np.zeros(0)}
                rows.append(selected_row(row,variant,values["fused"]))
        return rows,time.monotonic()-started
    long_rows,long_latency=evaluate(long_eval,"long_rvos");ground_rows,ground_latency=evaluate(ground_eval,"groundmore")
    add_prior_v1(long_rows,Path(args.v1_long_results),"long_rvos");add_prior_v1(ground_rows,Path(args.v1_ground_results),"groundmore")
    write_csv(root/"per_expression.csv",long_rows);write_csv(root/"groundmore_sequential.csv",ground_rows)
    long_summary=sum((summarize(long_rows,kind) for kind in ("static","dynamic","hybrid","explicit_order")),[])
    ground_summary=summarize(ground_rows,"sequential")+summarize(ground_rows,"sequential",True)
    write_csv(root/"long_summary.csv",long_summary);write_csv(root/"ground_summary.csv",ground_summary)
    order_rows=order_diagnostics(model_sets["opg_v2_full"],base_models,long_eval+ground_eval,args.device);write_csv(root/"order_diagnostics.csv",order_rows)
    diagnostics=diagnostic_summary(order_rows)
    comparisons={kind:{field:paired_bootstrap(long_rows,"opg_v2_full","mean_pool",kind,field) for field in ("selection_correct","J_and_F")} for kind in ("static","dynamic","hybrid","explicit_order")}
    ground_comparison={field:paired_bootstrap(ground_rows,"opg_v2_full","mean_pool","sequential",field,True) for field in ("selection_correct","J_and_F")}
    transitions={kind:transition_counts(long_rows,kind) for kind in ("static","dynamic","hybrid","explicit_order")}
    summary={
        "trainable_parameters":count_trainable_parameters(model_sets["opg_v2_full"][0]),"seeds":args.seeds,"training":training,
        "long_train_expressions":len(long_train_all),"long_train_hits":len(long_train_hits),"ground_train_expressions":len(ground_train_all),"ground_train_hits":len(ground_train_hits),
        "long_evaluation_expressions":len(long_eval),"ground_evaluation_expressions":len(ground_eval),
        "long_summary":long_summary,"ground_summary":ground_summary,"comparisons":comparisons,"ground_hit_comparison":ground_comparison,
        "order_diagnostics":diagnostics,"transitions":transitions,"latency_seconds":{"long_rvos":long_latency,"groundmore":ground_latency},
        "final_beta_order":[x["final_beta_order"] for x in training if x["variant"]=="opg_v2_full"],
        "final_beta_delta":[x["final_beta_delta"] for x in training if x["variant"]=="opg_v2_full"],
        "ground_truth_used_for_inference":False,
    }
    atomic_json(root/"summary.json",summary);atomic_json(root/"STATUS.json",{"state":"complete","phase":"evaluation_complete","pid":os.getpid(),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("train_features","train_metrics","eval_features","eval_metrics","mean_checkpoints","v1_checkpoints","v1_long_results","v1_ground_results","ground_train_features","ground_train_metrics","ground_eval_features","ground_eval_metrics","output"):p.add_argument(f"--{name.replace('_','-')}",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--seeds",type=int,nargs="+",default=[11,23,42]);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--epochs",type=int,default=40);p.add_argument("--patience",type=int,default=6);p.add_argument("--margin",type=float,default=.2);p.add_argument("--hit-threshold",type=float,default=.3);return p.parse_args()


if __name__=="__main__":run(parse_args())

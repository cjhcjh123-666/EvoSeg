"""Measure why the trained OPG-v1 residual did not alter candidate decisions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.evoseg.factorized_temporal_spatial.protocol import load_mean_pool
from projects.evoseg.ordered_process_grounding.protocol import is_order_sensitive
from projects.evoseg.ordered_process_grounding.train_sft import base_logits, load_ground_base, load_model, opg_examples, order_inputs
from projects.evoseg.temporal_grounding_mechanism.common import write_csv


def margin(values: np.ndarray, target: int | None) -> float | None:
    if target is None or len(values)<2:return None
    return float(values[target]-np.delete(values,target).max())


def audit_dataset(rows,models,bases,device,dataset):
    import torch
    records=[];expressions=[]
    with torch.inference_mode():
        for row in rows:
            if not row["tracks"].shape[0]: continue
            per_base=[];per_order=[];per_residual=[]
            query,tracks=order_inputs(row,device)
            alphas=[]
            for model,base in zip(models,bases):
                base_value=base_logits(base,row,device).float().cpu().numpy()
                order=model.order_scores(query,tracks)[0][0].float().cpu().numpy()
                alpha=float(torch.sigmoid(model.alpha).cpu());alphas.append(alpha)
                per_base.append(base_value);per_order.append(order);per_residual.append(alpha*order)
            base=np.mean(per_base,axis=0);order=np.mean(per_order,axis=0);residual=np.mean(per_residual,axis=0);final=base+residual
            base_std=float(np.std(base));order_std=float(np.std(order));residual_std=float(np.std(residual));ratio=residual_std/(base_std+1e-12)
            top_base=int(np.argmax(base));top_final=int(np.argmax(final));target=row.get("target_index")
            group="groundmore" if dataset=="groundmore" else "explicit_order" if is_order_sensitive(row["expression"]) else row["description_type"]
            expressions.append({"identity":row["identity"],"dataset":dataset,"group":group,"video_id":row["video_id"],"object_id":row["object_id"],
                                "expression_id":row["expression_id"],"candidate_count":len(base),"candidate_hit":int(row["candidate_hit"]),"alpha":float(np.mean(alphas)),
                                "base_score_std":base_std,"order_score_std":order_std,"effective_residual_std":residual_std,"residual_base_scale_ratio":ratio,
                                "top1_flipped":int(top_base!=top_final),"target_margin_base":margin(base,target),"target_margin_final":margin(final,target)})
            ids=[int(value) for value in row["candidate_track_ids"]]
            for index,track_id in enumerate(ids):
                records.append({"identity":row["identity"],"dataset":dataset,"group":group,"video_id":row["video_id"],"object_id":row["object_id"],
                                "expression_id":row["expression_id"],"candidate_track_id":track_id,"is_target":int(target==index) if target is not None else 0,
                                "base_score":float(base[index]),"order_score":float(order[index]),"effective_residual":float(residual[index]),"final_score":float(final[index]),
                                "alpha":float(np.mean(alphas)),"base_score_std":base_std,"order_score_std":order_std,"effective_residual_std":residual_std,
                                "residual_base_scale_ratio":ratio,"top1_flipped":int(top_base!=top_final),"target_margin_base":margin(base,target),"target_margin_final":margin(final,target)})
    return records,expressions


def summarize(expressions):
    output={}
    for group in ("static","dynamic","hybrid","explicit_order","groundmore"):
        subset=[row for row in expressions if row["group"]==group]
        if not subset:continue
        output[group]={"expressions":len(subset),"candidate_hits":sum(row["candidate_hit"] for row in subset),
                       "alpha_mean":float(np.mean([row["alpha"] for row in subset])),
                       "base_score_std":float(np.mean([row["base_score_std"] for row in subset])),
                       "order_score_std":float(np.mean([row["order_score_std"] for row in subset])),
                       "effective_residual_std":float(np.mean([row["effective_residual_std"] for row in subset])),
                       "residual_base_scale_ratio":float(np.mean([row["residual_base_scale_ratio"] for row in subset])),
                       "top1_flip_rate":float(np.mean([row["top1_flipped"] for row in subset])),
                       "target_margin_base":float(np.mean([row["target_margin_base"] for row in subset if row["target_margin_base"] is not None])),
                       "target_margin_final":float(np.mean([row["target_margin_final"] for row in subset if row["target_margin_final"] is not None]))}
    return output


def run(args):
    device=args.device;seeds=(11,23,42);checkpoint=Path(args.v1_checkpoints)
    models=[load_model(checkpoint/f"opg_full_seed{seed}.pt",device)[0] for seed in seeds]
    long_bases=[load_mean_pool(Path(args.mean_checkpoints)/f"mean_pool_seed{seed}.pt",device)[0] for seed in seeds]
    ground_bases=[load_ground_base(checkpoint/f"ground_mean_pool_seed{seed}.pt",device)[0] for seed in seeds]
    long_rows,_=opg_examples(Path(args.long_features),Path(args.long_metrics),.3,"concept")
    ground_rows,_=opg_examples(Path(args.ground_features),Path(args.ground_metrics),.3,"raw_expression")
    for row in long_rows:row["_base_kind"]="long_rvos"
    for row in ground_rows:row["_base_kind"]="groundmore"
    long_records,long_expressions=audit_dataset(long_rows,models,long_bases,device,"long_rvos")
    ground_records,ground_expressions=audit_dataset(ground_rows,models,ground_bases,device,"groundmore")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);write_csv(output/"v1_score_audit.csv",long_records+ground_records)
    summary=summarize(long_expressions+ground_expressions);(output/"v1_score_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# OPG-v1 score audit","","The audit reuses the saved three-seed v1 ensemble and frozen evaluation features; no model is retrained.","",
           "| Subset | expressions | base std | order std | effective residual std | residual/base ratio | top-1 flip | margin before | margin after |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,value in summary.items():
        lines.append(f"| {key} | {value['expressions']} | {value['base_score_std']:.5f} | {value['order_score_std']:.5f} | {value['effective_residual_std']:.5f} | {value['residual_base_scale_ratio']:.5f} | {value['top1_flip_rate']*100:.2f}% | {value['target_margin_base']:.5f} | {value['target_margin_final']:.5f} |")
    (output/"V1_SCORE_AUDIT.md").write_text("\n".join(lines)+"\n")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("v1_checkpoints","mean_checkpoints","long_features","long_metrics","ground_features","ground_metrics","output"):p.add_argument(f"--{name.replace('_','-')}",required=True)
    p.add_argument("--device",default="cuda:0");return p.parse_args()


if __name__=="__main__":run(parse_args())

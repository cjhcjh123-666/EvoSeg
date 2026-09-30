"""Create lightweight OPG reports, CSVs, figures, and the SFT decision."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle: return list(csv.DictReader(handle))


def fmt(value, scale=1.0, digits=2):
    return "N/A" if value is None else f"{float(value)*scale:.{digits}f}"


def ci(value: dict, scale=1.0):
    return f"{fmt(value.get('mean'),scale)} [{fmt(value.get('ci_low'),scale)}, {fmt(value.get('ci_high'),scale)}]"


def lookup(summary: list[dict], kind: str, method: str):
    return next(row for row in summary if row["description_type"]==kind and row["method"]==method)


def candidate_runtime(root: Path):
    records={}
    for path in sorted((root/"groundmore_candidates").glob("shard-*/candidate_records.jsonl")):
        with path.open() as handle:
            for line in handle:
                row=json.loads(line)
                if row.get("prompt_method")!="raw_expression":continue
                key=row.get("key") or "/".join((row["dataset"],row["video_id"],str(row["object_id"]),str(row["expression_id"]),row["prompt_method"]))
                records[key]=row
    success=[row for row in records.values() if row.get("status")=="success" and row.get("latency_seconds_synchronized") is not None]
    latencies=[float(row["latency_seconds_synchronized"]) for row in success]
    peaks=[int(row.get("peak_memory_bytes",0)) for row in success]
    return {
        "records":len(records),"success":len(success),"failed":len(records)-len(success),
        "mean_latency_seconds":float(np.mean(latencies)) if latencies else None,
        "median_latency_seconds":float(np.median(latencies)) if latencies else None,
        "p95_latency_seconds":float(np.quantile(latencies,.95)) if latencies else None,
        "peak_memory_bytes":max(peaks,default=0),
    }


def transition_counts(rows: list[dict]):
    paired=defaultdict(dict)
    for row in rows:
        if row["seed"]=="ensemble" and row["method"] in {"mean_pool","opg_full"}:
            paired[row["identity"]][row["method"]]=row
    counts={"paired":0,"candidate_miss":0,"opg_fixes_mean":0,"opg_damages_mean":0,"both_correct":0,"both_wrong":0}
    for methods in paired.values():
        if len(methods)!=2:continue
        counts["paired"]+=1
        if not int(methods["opg_full"]["candidate_hit"]):counts["candidate_miss"]+=1
        mean_ok=int(methods["mean_pool"]["selection_correct"]);opg_ok=int(methods["opg_full"]["selection_correct"])
        if opg_ok and not mean_ok:counts["opg_fixes_mean"]+=1
        elif mean_ok and not opg_ok:counts["opg_damages_mean"]+=1
        elif mean_ok and opg_ok:counts["both_correct"]+=1
        else:counts["both_wrong"]+=1
    return counts


def plot_bars(path: Path, title: str, labels: list[str], values: list[float], ylabel: str):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));colors=["#8da0cb","#66c2a5","#fc8d62","#e78ac3"][:len(values)]
    bars=ax.bar(labels,values,color=colors);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.2)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value,f"{value:.2f}",ha="center",va="bottom")
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def qualitative(root: Path, docs: Path, manifest_path: Path, rows: list[dict], order_rows: list[dict]):
    import matplotlib.pyplot as plt
    manifest=json.loads(manifest_path.read_text());items={(x["video_id"],str(x["object_id"])):x for x in manifest["objects"]}
    image_root=Path(manifest["dataset"]["image_root"]);ensemble=[x for x in rows if x["seed"]=="ensemble"]
    by_identity=defaultdict(dict)
    for row in ensemble:by_identity[row["identity"]][row["method"]]=row
    full_order={x["identity"]:x for x in order_rows if x["method"]=="opg_full"}
    success=[];failure=[]
    for identity,value in full_order.items():
        methods=by_identity.get(identity,{})
        if "opg_full" not in methods:continue
        score=min(float(value["original_minus_reverse"]),float(value["original_minus_block_swap"]))
        target=(score,identity)
        if int(methods["opg_full"]["selection_correct"]) and score>0:success.append(target)
        else:failure.append(target)
    success=sorted(success,reverse=True)[:3];failure=sorted(failure)[:3]
    static=[]
    for identity,methods in by_identity.items():
        if methods.get("opg_full",{}).get("description_type")=="static" and int(methods["opg_full"]["selection_correct"]) and int(methods.get("mean_pool",{}).get("selection_correct",0)):
            static.append((0.0,identity))
    static=sorted(static,key=lambda x:x[1])[:3]
    misses=[]
    for identity,methods in by_identity.items():
        if "opg_full" in methods and not int(methods["opg_full"]["candidate_hit"]):misses.append((0.0,identity))
    misses=sorted(misses,key=lambda x:x[1])[:3]
    categories={"order_success":success,"bag_of_frames_failure":failure,"static_cue":static,"candidate_generation_failure":misses}
    target_dir=docs/"figures"/"qualitative_order_cases";target_dir.mkdir(parents=True,exist_ok=True)
    audit=[]
    for category,values in categories.items():
        for _score,identity in values:
            parts=identity.split("/");item=items[(parts[1],parts[2])];frame_indices=np.linspace(0,item["frame_count"]-1,num=4,dtype=int)
            fig,axes=plt.subplots(1,4,figsize=(12,3))
            for ax,index in zip(axes,frame_indices):
                ax.imshow(plt.imread(image_root/item["video_id"]/f"{item['frame_names'][index]}.jpg"));ax.set_title(f"frame {index}");ax.axis("off")
            query=by_identity[identity]["opg_full"]["expression"]
            margin=full_order.get(identity,{}).get("original_minus_reverse")
            fig.suptitle(f"{category} | reverse margin={fmt(margin,1,3)}\n{query}",fontsize=9);fig.tight_layout()
            path=target_dir/f"{category}__{identity.replace('/','__')}.png";fig.savefig(path,dpi=150);plt.close(fig)
            audit.append({"category":category,"identity":identity,"selection_rule":"fixed deterministic extreme/rank rule","path":str(path.relative_to(docs))})
    (docs/"qualitative_cases.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")


def run(args) -> None:
    artifact=Path(args.artifact).resolve();docs=Path(args.docs).resolve();docs.mkdir(parents=True,exist_ok=True);(docs/"figures").mkdir(exist_ok=True)
    summary=json.loads((artifact/"summary.json").read_text());long=summary["summary"];ground=summary["groundmore_summary"]
    ground_train_manifest=json.loads((artifact/"groundmore_trainval_sequential_manifest.json").read_text())
    ground_test_manifest=json.loads((artifact/"groundmore_test_sequential_manifest.json").read_text())
    ground_train_failures=ground_train_manifest.get("failures",[]);ground_test_failures=ground_test_manifest.get("failures",[])
    comparisons=summary["comparisons"];ground_comparison=summary["groundmore_comparison"];order=summary["groundmore_order_summary"]
    dynamic_mean=lookup(long,"dynamic","mean_pool");dynamic_full=lookup(long,"dynamic","opg_full")
    dynamic_static=lookup(long,"dynamic","static_identity");dynamic_bigru=lookup(long,"dynamic","bigru_ftsg")
    static_mean=lookup(long,"static","mean_pool");static_full=lookup(long,"static","opg_full")
    ground_mean=lookup(ground,"sequential","mean_pool");ground_no=lookup(ground,"sequential","opg_no_order_loss");ground_full=lookup(ground,"sequential","opg_full")
    reverse=order["opg_full"]["original_minus_reverse"];block=order["opg_full"]["original_minus_block_swap"]
    sensitivity_better=(order["opg_full"]["reverse_sensitivity_rate"]>order["opg_no_order_loss"]["reverse_sensitivity_rate"] and order["opg_full"]["block_swap_sensitivity_rate"]>order["opg_no_order_loss"]["block_swap_sensitivity_rate"])
    full_go=(ground_comparison["J_and_F"]["ci_low"]>0 and ground_comparison["selection_accuracy"]["ci_low"]>0 and reverse["ci_low"]>0 and block["ci_low"]>0 and sensitivity_better and comparisons["dynamic"]["J_and_F"]["mean"]>=-0.005 and comparisons["static"]["J_and_F"]["mean"]>=-0.02)
    partial=(not full_go and ground_comparison["J_and_F"]["mean"]>0 and reverse["mean"]>0 and block["mean"]>0 and comparisons["dynamic"]["J_and_F"]["mean"]>=-0.01)
    decision="GO" if full_go else "PARTIAL GO" if partial else "NO-GO"

    runtime_rows=read_csv(artifact/"runtime.csv")
    candidate_cost=candidate_runtime(artifact)
    long_rows=read_csv(artifact/"per_expression.csv")
    ground_rows=read_csv(artifact/"groundmore_sequential.csv")
    long_transitions=transition_counts(long_rows);ground_transitions=transition_counts(ground_rows)

    for name in ("per_expression.csv","order_pairs.csv","groundmore_sequential.csv","runtime.csv"):
        shutil.copy2(artifact/name,docs/name)
    with (docs/"ablations.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list((long+ground)[0]));writer.writeheader();writer.writerows(long+ground)

    plot_bars(docs/"figures"/"dynamic_results.png","Long-RVOS Dynamic candidate-direct J&F",["Mean pool","No monotonic","No order loss","Full OPG"],[lookup(long,"dynamic",x)["J_and_F"]*100 for x in ("mean_pool","opg_no_monotonic","opg_no_order_loss","opg_full")],"J&F")
    plot_bars(docs/"figures"/"normal_vs_permuted.png","GroundMoRe target process score",["Original","Reverse","Block swap"],[0, -reverse["mean"],-block["mean"]],"Score relative to original")
    plot_bars(docs/"figures"/"order_margin.png","GroundMoRe order margin",["No order loss: rev","Full: rev","Full: block"],[order["opg_no_order_loss"]["original_minus_reverse"]["mean"],reverse["mean"],block["mean"]],"Original - permutation")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,3));ax.axis("off")
    labels=[("Frozen query tokens",.08),("4 process slots\n+ ordinal + gates",.31),("Monotonic soft DP\norder residual",.55),("Mean-pool base\n+ α·order",.76),("Selected SAM3.1\ncandidate mask",.94)]
    for text,x in labels:ax.text(x,.5,text,ha="center",va="center",bbox=dict(boxstyle="round",fc="#eef4ff",ec="#315d9c"),transform=ax.transAxes)
    for (_,a),(_,b) in zip(labels,labels[1:]):ax.annotate("",xy=(b-.07,.5),xytext=(a+.07,.5),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    fig.tight_layout();fig.savefig(docs/"figures"/"method_overview.png",dpi=180);plt.close(fig)
    qualitative(artifact,docs,Path(args.long_manifest),long_rows,read_csv(artifact/"order_pairs.csv"))

    method=f"""# Ordered Process Grounding (OPG) SFT

OPG preserves the frozen order-agnostic multi-frame identity score and adds only a learned residual: `s_final = s_base + sigmoid(alpha) * s_order`. Four learned process slots cross-attend the untouched official query-token sequence, receive ordinal embeddings and validity gates, and align to eight ordered frozen candidate-region features through a differentiable log-sum-exp over all strictly increasing paths. The selected frozen SAM3.1 candidate track is emitted directly; the order branch never predicts pixels.

- Trainable OPG parameters: {summary['trainable_parameters']:,}
- Frozen: Sa2VA, SAM3.1, candidate generator and feature encoders
- Order negative: exactly one identity-hashed reverse or half-block swap per verified sequential training sample
- Fixed loss: `L_id + L_order`, margin 0.2, lambda 1.0
"""
    (docs/"METHOD.md").write_text(method)
    (docs/"DATA_AUDIT.md").write_text(f"""# Data audit

- Long-RVOS official-train cache: {summary['train_expressions']} expressions, {summary['train_candidate_hits']} candidate hits; deterministic connector filter yields {summary['order_fit_expressions']} fitting expressions over {summary['order_fit_videos']} videos.
- Long-RVOS paired validation: {summary['evaluation_expressions']} expressions; {summary['evaluation_order_expressions']} deterministic order-word hits.
- GroundMoRe source: official author repository commit `d5074ab920a86a7ea92ecc69902109f6887f3e10` and official v2 metadata. Trainval has {ground_train_manifest['selection']['official_sequential_expressions']} official Sequential expressions: {summary['groundmore_train_expressions']} available and {len(ground_train_failures)} explicitly retained metadata/archive failures. Held-out official test has {ground_test_manifest['selection']['official_sequential_expressions']} expressions: {summary['groundmore_eval_expressions']} available and {len(ground_test_failures)} failures.
- Classification uses only GroundMoRe's official `q_type=Sequential` or the preregistered Long-RVOS connector regex. No LLM/manual event labels, new queries, or pseudo labels were used.
- GT is used only for candidate-to-target assignment and evaluation, never candidate generation or inference scoring.
""")
    seed_lines="\n".join(f"- {x['variant']} seed {x['seed']}: {x['elapsed_seconds']:.1f}s, alpha={x['final_alpha']:.4f}, params={x['trainable_parameters']:,}" for x in summary["training"])
    (docs/"TRAINING_REPORT.md").write_text(f"# Training report\n\nSeeds are fixed at 11/23/42; no final-validation seed selection or hyperparameter sweep was performed. Stage 1 is exactly one identity-only epoch, followed by fixed-budget ordered SFT with train-internal source-video early stopping.\n\n{seed_lines}\n")
    (docs/"LONG_RVOS_RESULTS.md").write_text(f"""# Long-RVOS full paired validation

| Method | Dynamic accuracy | J | F | J&F |
|---|---:|---:|---:|---:|
| Static identity | {fmt(dynamic_static['selection_accuracy'],100)} | {fmt(dynamic_static['J'],100)} | {fmt(dynamic_static['F'],100)} | {fmt(dynamic_static['J_and_F'],100)} |
| Mean pool | {fmt(dynamic_mean['selection_accuracy'],100)} | {fmt(dynamic_mean['J'],100)} | {fmt(dynamic_mean['F'],100)} | {fmt(dynamic_mean['J_and_F'],100)} |
| BiGRU FTSG (reused) | {fmt(dynamic_bigru['selection_accuracy'],100)} | {fmt(dynamic_bigru['J'],100)} | {fmt(dynamic_bigru['F'],100)} | {fmt(dynamic_bigru['J_and_F'],100)} |
| Full OPG | {fmt(dynamic_full['selection_accuracy'],100)} | {fmt(dynamic_full['J'],100)} | {fmt(dynamic_full['F'],100)} | {fmt(dynamic_full['J_and_F'],100)} |

Full OPG − mean pool Dynamic J&F: {ci(comparisons['dynamic']['J_and_F'],100)} pp. Static J&F changes from {fmt(static_mean['J_and_F'],100)} to {fmt(static_full['J_and_F'],100)} ({fmt(comparisons['static']['J_and_F']['mean'],100)} pp).
""")
    (docs/"GROUNDMORE_RESULTS.md").write_text(f"""# GroundMoRe Sequential

| Method | Selection accuracy | J | F | J&F |
|---|---:|---:|---:|---:|
| Mean pool | {fmt(ground_mean['selection_accuracy'],100)} | {fmt(ground_mean['J'],100)} | {fmt(ground_mean['F'],100)} | {fmt(ground_mean['J_and_F'],100)} |
| OPG no order loss | {fmt(ground_no['selection_accuracy'],100)} | {fmt(ground_no['J'],100)} | {fmt(ground_no['F'],100)} | {fmt(ground_no['J_and_F'],100)} |
| Full OPG | {fmt(ground_full['selection_accuracy'],100)} | {fmt(ground_full['J'],100)} | {fmt(ground_full['F'],100)} | {fmt(ground_full['J_and_F'],100)} |

Full OPG − mean pool: accuracy {ci(ground_comparison['selection_accuracy'],100)} pp; J&F {ci(ground_comparison['J_and_F'],100)} pp (source-video cluster bootstrap, 2,000 resamples).
""")
    (docs/"ORDER_SENSITIVITY.md").write_text(f"""# Order sensitivity

- Full OPG Original − Reverse: {ci(reverse)}
- Full OPG Original − BlockSwap: {ci(block)}
- Sensitivity rates: reverse {fmt(order['opg_full']['reverse_sensitivity_rate'],100)}%, block swap {fmt(order['opg_full']['block_swap_sensitivity_rate'],100)}%
- No-order-loss rates: reverse {fmt(order['opg_no_order_loss']['reverse_sensitivity_rate'],100)}%, block swap {fmt(order['opg_no_order_loss']['block_swap_sensitivity_rate'],100)}%
- Mean-pool margins are exactly zero by construction and verified by the permutation-invariance test.

Permutation diagnostics keep the candidate, frame set, visual content, and query fixed. They change temporal order only; no alternative GT label is asserted.
""")
    (docs/"ABLATIONS.md").write_text("# Ablations\n\n`ablations.csv` contains object-balanced Long-RVOS and GroundMoRe results for mean-pool, no-monotonic global attention, monotonic without order loss, and full monotonic OPG SFT. All new variants use identical frozen inputs and three fixed seeds.\n")
    selector_lines=[]
    for row in runtime_rows:
        seconds=float(row["latency_seconds"]);expressions=max(int(row["expressions"]),1)
        selector_lines.append(f"| {row['dataset']} | {row['method']} | {seconds/expressions*1000:.3f} | {int(row['peak_memory_bytes'])/2**30:.3f} |")
    (docs/"EFFICIENCY.md").write_text(f"""# Efficiency

- OPG trainable parameters per seed: {summary['trainable_parameters']:,}; three-seed ensemble loaded parameters: {summary['trainable_parameters']*3:,}.
- GroundMoRe official SAM3.1 raw-expression candidate generation: {candidate_cost['success']}/{candidate_cost['records']} successful; synchronized mean/median/p95 latency {fmt(candidate_cost['mean_latency_seconds'],1,3)}/{fmt(candidate_cost['median_latency_seconds'],1,3)}/{fmt(candidate_cost['p95_latency_seconds'],1,3)} seconds per expression; observed peak {candidate_cost['peak_memory_bytes']/2**30:.2f} GiB.
- Candidate generation and selector timing are reported separately. Concurrently contended GPU wall time is not presented as a clean deployment benchmark.

| Dataset | component | synchronized ms/query | peak allocated GiB |
|---|---|---:|---:|
{chr(10).join(selector_lines)}
""")
    ground_order_rows=read_csv(artifact/"groundmore_order_pairs.csv")
    full_order=[row for row in ground_order_rows if row["method"]=="opg_full"]
    positive_reverse=sum(float(row["original_minus_reverse"])>0 for row in full_order)
    positive_block=sum(float(row["original_minus_block_swap"])>0 for row in full_order)
    (docs/"FAILURE_ANALYSIS.md").write_text(f"""# Failure analysis

Counts below follow fixed rules over every ensemble result; no cases were dropped based on outcome.

| Dataset | paired | candidate miss | OPG fixes mean-pool | OPG damages mean-pool | both correct | both wrong |
|---|---:|---:|---:|---:|---:|---:|
| Long-RVOS | {long_transitions['paired']} | {long_transitions['candidate_miss']} | {long_transitions['opg_fixes_mean']} | {long_transitions['opg_damages_mean']} | {long_transitions['both_correct']} | {long_transitions['both_wrong']} |
| GroundMoRe Sequential | {ground_transitions['paired']} | {ground_transitions['candidate_miss']} | {ground_transitions['opg_fixes_mean']} | {ground_transitions['opg_damages_mean']} | {ground_transitions['both_correct']} | {ground_transitions['both_wrong']} |

For verified GroundMoRe target tracks, Full OPG scores original above reverse in {positive_reverse}/{len(full_order)} cases and above block-swap in {positive_block}/{len(full_order)} cases. Candidate misses remain an executor ceiling and are always scored as failures. `figures/qualitative_order_cases/` contains deterministic success, bag-of-frames failure, static-cue, and candidate-miss examples rather than success-only curation.

Official metadata/archive availability failures are retained separately: trainval {len(ground_train_failures)}, test {len(ground_test_failures)}. They are not converted into positive candidates or silently counted as successful training samples.
""")
    (docs/"FINAL_SFT_GO_NOGO.md").write_text(f"""# Final SFT decision: {decision}

The preregistered gate is applied without post-hoc tuning. GroundMoRe J&F delta is {ci(ground_comparison['J_and_F'],100)} pp; reverse and block-swap order margins are {ci(reverse)} and {ci(block)}. Long-RVOS Dynamic J&F delta is {ci(comparisons['dynamic']['J_and_F'],100)} pp and Static delta is {ci(comparisons['static']['J_and_F'],100)} pp.

{'The SFT gate passes. RL is not run; only a constrained design is documented.' if decision=='GO' else 'The SFT gate does not fully pass. RL is not designed or run; the result remains an SFT diagnostic.'}
""")
    if decision=="GO":
        (docs/"RL_DESIGN.md").write_text("""# RL design (not executed)

Start exclusively from the frozen-backbone OPG SFT checkpoint and retain the monotonic architecture. Optimize three separately logged rewards: `R_identity` for correct candidate selection, `R_mask` for the selected frozen candidate's official mask J&F, and `R_order` only on verified Sequential samples for increasing the correct-target original-vs-fixed-permutation decision margin. The objective is to suppress bag-of-frames shortcuts after SFT—not to learn time from scratch. No RL job has been started.
""")
    atomic={"decision":decision,"long_dynamic_delta":comparisons["dynamic"],"groundmore_delta":ground_comparison,"order":order}
    (artifact/"FINAL_DECISION.json").write_text(json.dumps(atomic,indent=2)+"\n")
    print(json.dumps(atomic,indent=2))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifact",required=True);p.add_argument("--docs",required=True);p.add_argument("--long-manifest",required=True);return p.parse_args()


if __name__=="__main__":run(parse_args())

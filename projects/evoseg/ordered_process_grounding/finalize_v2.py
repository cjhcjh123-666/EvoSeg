"""Create lightweight OPG-v2 reports from immutable experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def lookup(rows,subset,method,hit=0):
    return next(row for row in rows if row["subset"]==subset and row["method"]==method and int(row["hit_only"])==hit)


def pc(value):return f"{float(value)*100:.2f}"
def interval(value,scale=100):return f"{float(value['mean'])*scale:+.2f} [{float(value['ci_low'])*scale:+.2f}, {float(value['ci_high'])*scale:+.2f}]"


def read_csv(path):
    with path.open() as handle:return list(csv.DictReader(handle))


def copy(path,destination):
    if path.is_file():shutil.copy2(path,destination/path.name)


def bars(path,title,labels,values,ylabel):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));items=ax.bar(labels,values,color=["#779ecb","#f6a65a","#66b38e","#b27db8"][:len(values)])
    ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.2)
    for item,value in zip(items,values):ax.text(item.get_x()+item.get_width()/2,value,f"{value:.2f}",ha="center",va="bottom" if value>=0 else "top")
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def run(args):
    artifact=Path(args.artifact);docs=Path(args.docs);docs.mkdir(parents=True,exist_ok=True);figures=docs/"figures";figures.mkdir(exist_ok=True)
    summary=json.loads((artifact/"summary.json").read_text());audit=json.loads((artifact/"v1_score_audit.json").read_text())
    candidate_root=artifact/"groundmore_concept_candidates";candidate=json.loads((candidate_root/"candidate_generation_audit.json").read_text())
    long=summary["long_summary"];ground=summary["ground_summary"];comparison=summary["comparisons"];ground_cmp=summary["ground_hit_comparison"];order=summary["order_diagnostics"]
    explicit_base=lookup(long,"explicit_order","mean_pool");explicit_full=lookup(long,"explicit_order","opg_v2_full")
    dynamic_base=lookup(long,"dynamic","mean_pool");dynamic_full=lookup(long,"dynamic","opg_v2_full")
    static_base=lookup(long,"static","mean_pool");static_full=lookup(long,"static","opg_v2_full")
    ground_base=lookup(ground,"sequential","mean_pool",1);ground_v1=lookup(ground,"sequential","opg_v1",1);ground_full=lookup(ground,"sequential","opg_v2_full",1)
    ground_order=lookup(ground,"sequential","opg_v2_order_only",1);ground_delta=lookup(ground,"sequential","opg_v2_delta_only",1)
    ground_all=lookup(ground,"sequential","opg_v2_full",0)
    reverse=order["long_rvos"]["reverse"];block=order["long_rvos"]["block_swap"]
    explicit_positive=comparison["explicit_order"]["selection_correct"]["ci_low"]>0 or (comparison["explicit_order"]["selection_correct"]["mean"]>=.02 and comparison["explicit_order"]["J_and_F"]["mean"]>=.01)
    ground_positive=ground_cmp["selection_correct"]["mean"]>0
    relative_positive=reverse["relative_order_margin"]["ci_low"]>0 and block["relative_order_margin"]["ci_low"]>0
    stable=comparison["dynamic"]["J_and_F"]["mean"]>=-.01 and comparison["static"]["J_and_F"]["mean"]>=-.02
    full_go=explicit_positive and ground_positive and relative_positive and stable
    partial=(not full_go and comparison["explicit_order"]["selection_correct"]["mean"]>0 and ground_cmp["selection_correct"]["mean"]>=0 and reverse["relative_order_margin"]["mean"]>0 and block["relative_order_margin"]["mean"]>0 and stable)
    decision="FULL GO" if full_go else "PARTIAL GO" if partial else "NO-GO"
    for name in ("v1_score_audit.csv","per_expression.csv","groundmore_sequential.csv","order_diagnostics.csv","long_summary.csv","ground_summary.csv"):
        copy(artifact/name,docs)
    copy(candidate_root/"candidate_generation_audit.csv",docs);copy(artifact/"V1_SCORE_AUDIT.md",docs);copy(candidate_root/"CANDIDATE_GENERATION_AUDIT.md",docs)
    (docs/"METHOD.md").write_text("""# Discriminative Ordered Process Grounding (OPG-v2)

OPG-v2 retains the four process slots, validity gates, frozen T=8 candidate features, and monotonic log-sum-exp alignment from v1. It adds candidate-order CE, candidate-relative order CE, and target self-order margin supervision. Candidate-wise standardized base, original-order, and original-minus-permutation logits are fused with positive learned `beta_order` and `beta_delta`, both initialized to 1. Frozen SAM3.1 candidate masks remain the spatial output.

No VLM, SAM3.1, candidate generator, or visual encoder parameter is updated. No artificial query, event label, GT inference prompt, RL, Agent, or CoT is used.
""")
    seed_lines=[]
    for value in summary["training"]:
        if value["variant"]=="opg_v2_full":seed_lines.append(f"- seed {value['seed']}: {value['elapsed_seconds']:.1f}s; beta_order={value['final_beta_order']:.4f}; beta_delta={value['final_beta_delta']:.4f}; params={value['trainable_parameters']:,}")
    (docs/"TRAINING_REPORT.md").write_text("# OPG-v2 training\n\nFixed seeds 11/23/42, v1 learning rate/split/epoch budget/patience, one identity warm-up epoch, and no final-validation seed selection or lambda sweep.\n\n"+"\n".join(seed_lines)+"\n")
    (docs/"LONG_RVOS_RESULTS.md").write_text(f"""# Long-RVOS results

| subset | method | selection accuracy | J | F | J&F |
|---|---|---:|---:|---:|---:|
| Explicit-order | Mean pool | {pc(explicit_base['selection_accuracy'])} | {pc(explicit_base['J'])} | {pc(explicit_base['F'])} | {pc(explicit_base['J_and_F'])} |
| Explicit-order | Full OPG-v2 | {pc(explicit_full['selection_accuracy'])} | {pc(explicit_full['J'])} | {pc(explicit_full['F'])} | {pc(explicit_full['J_and_F'])} |
| Dynamic | Mean pool | {pc(dynamic_base['selection_accuracy'])} | {pc(dynamic_base['J'])} | {pc(dynamic_base['F'])} | {pc(dynamic_base['J_and_F'])} |
| Dynamic | Full OPG-v2 | {pc(dynamic_full['selection_accuracy'])} | {pc(dynamic_full['J'])} | {pc(dynamic_full['F'])} | {pc(dynamic_full['J_and_F'])} |
| Static | Mean pool | {pc(static_base['selection_accuracy'])} | {pc(static_base['J'])} | {pc(static_base['F'])} | {pc(static_base['J_and_F'])} |
| Static | Full OPG-v2 | {pc(static_full['selection_accuracy'])} | {pc(static_full['J'])} | {pc(static_full['F'])} | {pc(static_full['J_and_F'])} |

- Explicit-order Full−Mean selection: {interval(comparison['explicit_order']['selection_correct'])} pp; J&F: {interval(comparison['explicit_order']['J_and_F'])} pp.
- Dynamic Full−Mean J&F: {interval(comparison['dynamic']['J_and_F'])} pp.
- Static Full−Mean J&F: {interval(comparison['static']['J_and_F'])} pp.
""")
    (docs/"GROUNDMORE_RESULTS.md").write_text(f"""# GroundMoRe Sequential

## Candidate-hit reasoning subset (55 expressions)

| selector | selection accuracy | candidate-direct J&F |
|---|---:|---:|
| Mean pool | {pc(ground_base['selection_accuracy'])} | {pc(ground_base['J_and_F'])} |
| OPG-v1 | {pc(ground_v1['selection_accuracy'])} | {pc(ground_v1['J_and_F'])} |
| OPG-v2 order-original-only | {pc(ground_order['selection_accuracy'])} | {pc(ground_order['J_and_F'])} |
| OPG-v2 delta-only | {pc(ground_delta['selection_accuracy'])} | {pc(ground_delta['J_and_F'])} |
| Full OPG-v2 | {pc(ground_full['selection_accuracy'])} | {pc(ground_full['J_and_F'])} |

Full−Mean selection: {interval(ground_cmp['selection_correct'])} pp; J&F: {interval(ground_cmp['J_and_F'])} pp.

## End-to-end all 480

Full OPG-v2 J/F/J&F = {pc(ground_all['J'])}/{pc(ground_all['F'])}/{pc(ground_all['J_and_F'])}. The raw-expression bank contains a target candidate at oracle J&F≥0.3 for 55/480 expressions; misses are retained at zero. The separate deterministic concept audit is not counted as OPG-v2 reasoning gain.
""")
    (docs/"ORDER_DIAGNOSTICS.md").write_text(f"""# Ordered ranking diagnostics

| dataset / permutation | self-order margin | candidate margin | relative-order margin | rank degradation |
|---|---:|---:|---:|---:|
| Long-RVOS / reverse | {interval(reverse['self_order_margin'],1)} | {interval(reverse['candidate_margin'],1)} | {interval(reverse['relative_order_margin'],1)} | {interval(reverse['rank_degradation'],1)} |
| Long-RVOS / block swap | {interval(block['self_order_margin'],1)} | {interval(block['candidate_margin'],1)} | {interval(block['relative_order_margin'],1)} | {interval(block['rank_degradation'],1)} |
| GroundMoRe / reverse | {interval(order['groundmore']['reverse']['self_order_margin'],1)} | {interval(order['groundmore']['reverse']['candidate_margin'],1)} | {interval(order['groundmore']['reverse']['relative_order_margin'],1)} | {interval(order['groundmore']['reverse']['rank_degradation'],1)} |
| GroundMoRe / block swap | {interval(order['groundmore']['block_swap']['self_order_margin'],1)} | {interval(order['groundmore']['block_swap']['candidate_margin'],1)} | {interval(order['groundmore']['block_swap']['relative_order_margin'],1)} | {interval(order['groundmore']['block_swap']['rank_degradation'],1)} |

Positive self-order margins on Long-RVOS confirm continued permutation sensitivity, but relative-order margins do not show that the target is more order-consistent than distractors.
""")
    methods=("mean_pool","opg_v1","v2_no_relative","v2_no_candidate","v2_no_monotonic","v2_no_delta_fusion","opg_v2_full")
    table=[]
    for method in methods:
        value=lookup(long,"explicit_order",method);table.append(f"| {method} | {pc(value['selection_accuracy'])} | {pc(value['J_and_F'])} |")
    (docs/"ABLATIONS.md").write_text("# Ablations\n\n| method | Explicit-order accuracy | Explicit-order J&F |\n|---|---:|---:|\n"+"\n".join(table)+"\n")
    transitions=summary["transitions"]
    (docs/"FAILURE_ANALYSIS.md").write_text("# Selection transitions\n\n| subset | fixes | damages | net fixes | both correct | both wrong |\n|---|---:|---:|---:|---:|---:|\n"+"\n".join(f"| {key} | {value['fixes']} | {value['damages']} | {value['net_fixes']} | {value['both_correct']} | {value['both_wrong']} |" for key,value in transitions.items())+"\n\nThe fixed rule reports all transitions; no outcome-based case deletion is used.\n")
    (docs/"FINAL_SFT_GO_NOGO.md").write_text(f"""# OPG-v2 SFT decision: {decision}

The fixed gate does not pass. Explicit-order point estimates improve, but selection CI crosses zero; GroundMoRe candidate-hit selection decreases; and target-relative order margins are non-positive. Betas remain near one, so this is not a repeat of v1 residual silencing. Candidate-relative ordered evidence has not become a stable target-vs-distractor signal.

No RL design is generated and no RL job is started.
""")
    bars(figures/"explicit_order_main.png","Long-RVOS explicit-order",["Mean accuracy","V2 accuracy","Mean J&F","V2 J&F"],[float(explicit_base['selection_accuracy'])*100,float(explicit_full['selection_accuracy'])*100,float(explicit_base['J_and_F'])*100,float(explicit_full['J_and_F'])*100],"percent")
    bars(figures/"groundmore_hit.png","GroundMoRe candidate-hit selection",["Mean","V1","Order","Delta","V2"],[float(x['selection_accuracy'])*100 for x in (ground_base,ground_v1,ground_order,ground_delta,ground_full)],"accuracy (%)")
    bars(figures/"relative_order_margin.png","Target-relative ordered evidence",["Long rev","Long block","Ground rev","Ground block"],[reverse['relative_order_margin']['mean'],block['relative_order_margin']['mean'],order['groundmore']['reverse']['relative_order_margin']['mean'],order['groundmore']['block_swap']['relative_order_margin']['mean']],"margin")
    final={"decision":decision,"explicit_order":comparison["explicit_order"],"dynamic":comparison["dynamic"],"groundmore_hit":ground_cmp,"order":order,"betas":{"order":summary["final_beta_order"],"delta":summary["final_beta_delta"]},"candidate_audit":candidate}
    (artifact/"FINAL_DECISION.json").write_text(json.dumps(final,indent=2)+"\n")
    print(json.dumps({"decision":decision},indent=2))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifact",required=True);p.add_argument("--docs",required=True);return p.parse_args()


if __name__=="__main__":run(parse_args())

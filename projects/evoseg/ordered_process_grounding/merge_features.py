"""Merge prediction-independent OPG frozen-feature shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run(args) -> None:
    import torch

    payload=[];records=[];identities=set();audits=[]
    for value in args.input:
        path=Path(value).resolve();part=torch.load(path,map_location="cpu");meta=json.loads(path.with_suffix(".json").read_text())
        for row,record in zip(part,meta["records"]):
            if row["identity"] != record["identity"]: raise RuntimeError(f"payload/metadata order mismatch: {path}")
            if row["identity"] in identities: raise RuntimeError(f"duplicate feature identity: {row['identity']}")
            identities.add(row["identity"]);payload.append(row);records.append(record)
        audits.append({key:meta.get(key) for key in ("source_features","source_features_sha256","manifest","track_steps","ground_truth_read","shard_index","num_shards","sharding_unit")})
    order=sorted(range(len(payload)),key=lambda index:payload[index]["identity"]);payload=[payload[index] for index in order];records=[records[index] for index in order]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output)
    output.with_suffix(".json").write_text(json.dumps({
        "command":[sys.executable,*sys.argv],"merged_shards":audits,"ground_truth_read":any(bool(x.get("ground_truth_read")) for x in audits),
        "track_steps":audits[0]["track_steps"],"records":records,
    },indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"records":len(records),"output":str(output)}))


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",nargs="+",required=True);parser.add_argument("--output",required=True);return parser.parse_args()


if __name__=="__main__":run(parse_args())

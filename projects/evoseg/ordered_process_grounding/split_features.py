"""Split a merged frozen-feature cache by its official split field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args):
    import torch
    source=Path(args.input);payload=torch.load(source,map_location="cpu");meta=json.loads(source.with_suffix(".json").read_text())
    records={row["identity"]:row for row in meta["records"]}
    for split,output_value in (("trainval",args.train_output),("test",args.test_output)):
        selected=[row for row in payload if records[row["identity"]]["split"]==split]
        selected_records=[records[row["identity"]] for row in selected]
        output=Path(output_value);output.parent.mkdir(parents=True,exist_ok=True);torch.save(selected,output)
        output.with_suffix(".json").write_text(json.dumps({**{key:value for key,value in meta.items() if key!="records"},"official_split":split,"records":selected_records},indent=2,ensure_ascii=False)+"\n")
        print(split,len(selected),output)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True);p.add_argument("--train-output",required=True);p.add_argument("--test-output",required=True);return p.parse_args()


if __name__=="__main__":run(parse_args())


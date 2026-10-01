"""Audit GroundMoRe Sequential templates without inventing event labels.

The official metadata provides one ``action_start/action_end`` interval for the
entire question.  This audit deliberately separates (1) whether the language
can be split deterministically from (2) whether that interval can be assigned
to one clause.  Shared intervals across questions with different answer-bearing
clauses are direct evidence that clause-level localization is not annotated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .groundmore_parser import parse_sequential_query


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(metadata: Path) -> tuple[dict, list[dict]]:
    source = json.loads(metadata.read_text())["videos"]
    rows: list[dict] = []
    interval_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for video_id, video in source.items():
        for expression_id, item in video["questions"].items():
            if item["q_type"].lower() != "sequential":
                continue
            parsed = parse_sequential_query(item["question"])
            row = {
                "video_id": video_id,
                "expression_id": str(expression_id),
                "question": item["question"],
                "answer": item.get("answer"),
                "object_id": item.get("obj_id"),
                "action_start": item["action_start"],
                "action_end": item["action_end"],
                **parsed.to_dict(),
            }
            rows.append(row)
            interval_groups[(video_id, item["action_start"], item["action_end"])].append(row)

    resolved = [row for row in rows if row["resolved"]]
    shared = [group for group in interval_groups.values() if len(group) > 1]
    shared_different_targets = [
        group
        for group in shared
        if len({row["target_clause"] for row in group if row["target_clause"]}) > 1
    ]
    shared_before_after = [
        group
        for group in shared
        if {row["connective"] for row in group} >= {"before", "after"}
    ]
    counts = Counter(row["connective"] or "unresolved" for row in rows)
    summary = {
        "metadata": str(metadata.resolve()),
        "metadata_sha256": sha256(metadata),
        "sequential_expressions": len(rows),
        "source_videos": len({row["video_id"] for row in rows}),
        "language_parse_resolved": len(resolved),
        "language_parse_unresolved": len(rows) - len(resolved),
        "connective_counts": dict(sorted(counts.items())),
        "unique_annotated_intervals": len(interval_groups),
        "shared_interval_groups": len(shared),
        "shared_interval_groups_with_different_target_clauses": len(shared_different_targets),
        "shared_interval_groups_with_before_and_after": len(shared_before_after),
        "interval_semantics": "whole queried process / official mask-validity interval",
        "clause_interval_supervision_safe": False,
        "decision": (
            "Do not use action_start/action_end as a target-clause interval. "
            "It is shared by distinct questions/answer clauses over the same process."
        ),
        "examples": [
            [
                {
                    "question": row["question"],
                    "target_clause": row["target_clause"],
                    "connective": row["connective"],
                    "action_start": row["action_start"],
                    "action_end": row["action_end"],
                }
                for row in group[:4]
            ]
            for group in shared_different_targets[:5]
        ],
    }
    return summary, rows


def write_report(summary: dict, path: Path) -> None:
    counts = summary["connective_counts"]
    examples = []
    for group in summary["examples"]:
        examples.append(
            "\n".join(
                f"- `{item['action_start']}–{item['action_end']}` "
                f"({item['connective']}): {item['question']}"
                for item in group
            )
        )
    text = f"""# GroundMoRe Sequential Query Audit

## Verified source

- Metadata: `{summary['metadata']}`
- SHA-256: `{summary['metadata_sha256']}`
- Sequential expressions: **{summary['sequential_expressions']}** across **{summary['source_videos']}** videos

## Deterministic language parsing

- Resolved by exactly one supported connective: **{summary['language_parse_resolved']}**
- Unresolved: **{summary['language_parse_unresolved']}**
- Connectives: `{json.dumps(counts, sort_keys=True)}`

The parser only splits an explicit surface template and reorders `after` into
chronological order. It does not invent event labels or inspect masks.

## Interval semantics

The official metadata supplies one `action_start/action_end` pair per question.
There are **{summary['unique_annotated_intervals']}** unique video/interval groups;
**{summary['shared_interval_groups']}** are shared by multiple Sequential questions,
and **{summary['shared_interval_groups_with_different_target_clauses']}** share the
same interval despite different answer-bearing clauses. **{summary['shared_interval_groups_with_before_and_after']}**
groups contain both `before` and `after` formulations.

Therefore the interval is safe as the official mask-validity/full queried-process
window, but **not** as a clause-specific action annotation. CPG will preserve it
for official segmentation/evaluation. Clause-specific `L_loc` is disabled unless
an official clause-level field is found; using this interval as such would create
a pseudo-label.

## Representative shared intervals

{chr(10).join(examples)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = audit(args.metadata)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(summary, args.report)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

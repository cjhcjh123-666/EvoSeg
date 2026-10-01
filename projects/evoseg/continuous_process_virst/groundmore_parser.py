"""Deterministic audit/parser for GroundMoRe Sequential questions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParsedSequentialQuery:
    resolved: bool
    connective: str | None
    surface_left: str | None
    surface_right: str | None
    chronological_clauses: tuple[str, ...]
    target_clause: str | None
    reason: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["chronological_clauses"] = list(self.chronological_clauses)
        return value


_PATTERNS = (
    ("before", re.compile(r"\s+before\s+", re.IGNORECASE)),
    ("after", re.compile(r"\s+after\s+", re.IGNORECASE)),
    ("prior_to", re.compile(r"\s+prior\s+to\s+", re.IGNORECASE)),
    ("then", re.compile(r"\s+(?:and\s+)?then\s+", re.IGNORECASE)),
)


def parse_sequential_query(question: str) -> ParsedSequentialQuery:
    text = " ".join(question.strip().rstrip("?").split())
    matches = []
    for name, pattern in _PATTERNS:
        found = list(pattern.finditer(text))
        matches.extend((item.start(), item.end(), name) for item in found)
    if len(matches) != 1:
        return ParsedSequentialQuery(
            False,
            None,
            None,
            None,
            (),
            None,
            "requires exactly one supported explicit temporal connective",
        )
    start, end, connective = matches[0]
    left, right = text[:start].strip(), text[end:].strip()
    if not left or not right:
        return ParsedSequentialQuery(False, connective, left, right, (), None, "empty clause")
    if connective in {"after"}:
        chronological = (right, left)
    else:
        chronological = (left, right)
    # The interrogative/main clause is the answer-bearing clause. This is a
    # syntactic anchor only; whether the dataset interval localizes that clause
    # is audited separately and is not assumed by this parser.
    return ParsedSequentialQuery(
        True,
        connective,
        left,
        right,
        chronological,
        left,
        "deterministic single-connective split",
    )

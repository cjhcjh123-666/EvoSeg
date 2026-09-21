"""Deterministic temporal sampling used by experiments A and B."""

from __future__ import annotations

from collections.abc import Iterable


def nested_temporal_indices(length: int, budgets: Iterable[int]) -> dict[int, list[int]]:
    """Return ordered, exactly nested samples that cover the complete video.

    Indices are added greedily at the point farthest from the selected set.  The
    two endpoints are selected first, so this is full-range sampling rather than
    prefix truncation.  Taking prefixes of one schedule makes every smaller
    budget an exact subset of every larger budget (when the video is long
    enough).  Ties are resolved toward the earlier frame.
    """
    budgets = sorted({int(b) for b in budgets})
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if not budgets or budgets[0] <= 0:
        raise ValueError(f"budgets must be positive, got {budgets}")

    target = min(length, budgets[-1])
    selected: list[int] = [0]
    if target > 1 and length > 1:
        selected.append(length - 1)

    while len(selected) < target:
        chosen = set(selected)
        candidate = max(
            (i for i in range(length) if i not in chosen),
            key=lambda i: (min(abs(i - j) for j in selected), -i),
        )
        selected.append(candidate)

    return {budget: sorted(selected[: min(length, budget)]) for budget in budgets}


def uniform_indices(length: int, limit: int) -> list[int]:
    """Select at most ``limit`` full-range indices, including both endpoints."""
    if length <= 0 or limit <= 0:
        raise ValueError(f"length and limit must be positive: {length=}, {limit=}")
    return nested_temporal_indices(length, [limit])[limit]


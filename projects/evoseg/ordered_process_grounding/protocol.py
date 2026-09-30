"""Deterministic OPG data and permutation protocol."""
from __future__ import annotations

import hashlib
import re


ORDER_PATTERN = re.compile(
    r"\b(before|after|then|first|finally|followed\s+by|subsequently)\b", re.IGNORECASE
)


def is_order_sensitive(expression: str) -> bool:
    return ORDER_PATTERN.search(expression) is not None


def fixed_order_negative(identity: str, length: int = 8) -> tuple[str, list[int]]:
    """Choose exactly one preregistered permutation from an identity hash."""
    if length % 2:
        raise ValueError("block swap requires an even sequence length")
    digest = hashlib.sha256(f"opg-order-negative-v1:{identity}".encode()).digest()
    if digest[0] % 2 == 0:
        return "reverse", list(reversed(range(length)))
    half = length // 2
    return "block_swap", list(range(half, length)) + list(range(half))


def reverse_order(length: int = 8) -> list[int]:
    return list(reversed(range(length)))


def block_swap_order(length: int = 8) -> list[int]:
    if length % 2:
        raise ValueError("block swap requires an even sequence length")
    half = length // 2
    return list(range(half, length)) + list(range(half))


# GroundMoRe results

## Capability-only observations

The fixed training capability subset contains 32 official Sequential
expressions from 31 source videos. Of the 64 combined Long-RVOS/GroundMoRe
examples, 61 had deterministically resolvable order templates. At the final
occurrence, all 61 scored the original sequence above their identity-hashed
reverse or half-block-swap permutation. The mean margin was 0.2984 (reverse:
0.2821 over 30; block swap: 0.3143 over 31).

Thirty GroundMoRe examples exposed two to five official instance trajectories
for legal training-only object discrimination. The target beat the best
distractor in 29/30 cases (96.67%); mean target-minus-best-distractor margin was
7.7189.

Interval localization is N/A. The official interval is a queried-process and
mask-validity interval shared by questions whose answer clauses differ; it is
not a target-clause interval. See `GROUNDMORE_QUERY_AUDIT.md`.

## Benchmark evaluation

No CPG GroundMoRe validation/test evaluation was run because the capability
gate failed. The reused official VIRST reference remains J=25.543, F=27.967,
J&F=26.755 on 480 Sequential test expressions. It is included only as the
frozen baseline reference; no CPG delta or confidence interval exists.

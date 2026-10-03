# FTG pilot findings log

This file records pilot outcomes, including negative evidence. Values are held-out
J&F percentages on a 48-expression development pilot (36 train, 12 validation),
not benchmark results.

## Initial implementation

The initial FTG state residual was randomly initialized and immediately entered
the prompt through a gate near 0.5. FTG versus Frame Prompt changed Static,
Dynamic, Hybrid, and MeViS by +3.11, +0.00, +10.97, and -4.25 points. Identity
Only was the strongest control at 31.16 overall versus FTG at 20.35. Qualitative
inspection showed the state residual pulling prompts toward salient wrong people.

## Identity-anchored residual

Zero-initializing the dynamic projection makes the initial FTG prompt exactly
equal to Identity Only, after which temporal corrections are learned. At seed 11,
FTG reached 34.42 overall versus 25.85 for Frame Prompt. All four groups improved
(+20.24 Static, +11.89 Dynamic, +10.45 Hybrid, +2.96 MeViS), but the gain was not
selective to dynamic groups and each Long-RVOS group contained only two held-out
expressions. At seed 23 the result reversed (21.43 versus 28.37), proving that the
small pilot was unstable.

Original/Shuffle/Reverse changed FTG overall J&F by only about 1.2 points at seed
23. This does not yet establish temporal-order reasoning.

## Frozen SAM3.1 query-selection audit

At seed 11, FTG's actual predicted-query J&F was 34.42 while its training-only
matched-query J&F was 71.41. Frame Prompt was 25.85 versus 66.11. Query-selection
accuracy was only 3.1% and 6.8%, respectively. Thus the official frozen mask
decoder contains a strong mask among its object queries, but its predicted query
ranking is the dominant bottleneck. Matched-query masks remain diagnostic only
and are never used as inference outputs.

## Selection-loss calibration

Weights 0.3, 1.0, and 3.0 were tested without changing inference. Increasing the
weight did not materially fix query accuracy. Weight 1.0 was selected for the
larger replication because it removed the FTG/Frame sign reversal across seeds:
24.22 versus 24.15 at seed 11 and 24.85 versus 21.75 at seed 23. This choice is
based on cross-seed stability, not the best single run; the seed-11 FTG maximum
was 34.55 at weight 0.3.

The next gate therefore increases public training coverage and optimization
steps while keeping Qwen, frozen SAM3.1, the interface, frame budget, and loss
weight fixed. No verifier, oracle selection, or post-hoc matcher is introduced.

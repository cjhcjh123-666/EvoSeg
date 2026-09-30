# Order-sensitivity analysis

Order Preference Margin is the monotonic alignment score on the original frame
order minus the score on the same frames after a fixed permutation. Confidence
intervals use 2,000 source-video cluster bootstrap resamples.

| Condition | Seed | Original-Reverse | 95% CI | Original-BlockSwap | 95% CI |
|---|---:|---:|---:|---:|---:|
| Full | 11 | +0.00235 | [-0.00039,+0.00493] | +0.00299 | [+0.00051,+0.00536] |
| Full | 23 | -0.00039 | [-0.00175,+0.00100] | +0.00007 | [-0.00146,+0.00161] |
| Full | 42 | -0.00403 | [-0.00632,-0.00148] | -0.00311 | [-0.00482,-0.00147] |
| No order loss | 11 | +0.00002 | [-0.00102,+0.00117] | -0.00055 | [-0.00195,+0.00082] |
| Global/non-monotonic | 11 | 0 | [0,0] | 0 | [0,0] |

For the per-expression arithmetic mean across the three Full seeds:

- Original-Reverse: -0.00069, 95% CI [-0.00222,+0.00094].
- Original-BlockSwap: -0.00002, 95% CI [-0.00125,+0.00115].

The pre-prompt residual does respond numerically to permutations (mean L2 about
0.095--0.109 for Full), but its direction is not learned consistently. A global
attention control has exactly zero alignment-score margin by construction while
still changing its pre-prompt frame residual under permutation. This separates
mere feature response from a correct preference for the original process.

Training diagnostics agree with validation: over the final 50 verified-order
updates, mean margins were +0.00221, +0.00058, and +0.00006 for seeds 11, 23,
and 42, while the margin loss remained about 0.198--0.200. The SFT budget did
not reliably solve the fixed 0.2 ranking objective.

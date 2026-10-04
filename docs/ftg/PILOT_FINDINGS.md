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

## Scaled 192-expression replication

The scaled run used 144 train and 48 video-disjoint validation expressions for
four epochs (576 updates) with selection weight 1.0. The small-pilot improvement
did not survive. FTG versus Frame Prompt changed Static/Dynamic/Hybrid/MeViS by
-2.77/-6.10/-3.15/+0.53 points at seed 11 and
-3.21/-3.94/+1.26/-4.77 points at seed 23. Overall FTG was 33.80 versus 35.54
and 33.07 versus 36.44. The identity-state hypothesis is therefore **not
supported by the current gated implementation**.

The negative result remains diagnostic rather than a decoder-capacity failure.
At seed 11, FTG's matched-query J&F was 83.55 versus 74.79 for Frame Prompt,
while actual query-selection accuracy was only 2.34% versus 4.04%. State Only
and ID+State without gate reached 84.39 and 83.49 matched-query J&F. The
factorized states contain useful masks, but the permutation-matched training
target does not align them with the query chosen at inference.

The next controlled test replaces permutation matching plus score-based
inference with a single predetermined official SAM query slot used identically
for supervision and inference. This is an interface alignment: it adds no
inference module, does not search candidates, and never uses ground truth to
choose an output.

## Fixed-query-slot test

The fixed-slot interface also failed the cross-seed gate. At seed 11, FTG
reached 19.21 overall J&F versus 7.36 for Frame Prompt, but Long-RVOS Dynamic
dropped by 15.38 points. At seed 23, FTG and Frame Prompt were effectively tied
at 18.03 and 18.00, with Dynamic again lower by 3.12 points. The direction of
the result is therefore inconsistent with the identity/state motivation.

Qualitative inspection exposed two initialization-dependent failure modes on
the same MeViS clip: one run predicted nearly empty masks, while another spread
foreground across the boat body and windshield instead of the small referred
person. State Only reached 18.87 overall at seed 11, but its category scores
were similarly uneven (25.00 Static, 34.38 Dynamic, and 3.12 Hybrid). A fixed
slot removes permutation mismatch but does not provide a stable semantic
mapping from the learned prompt to that slot.

The score-head audit also identified a loss-semantics mismatch. SAM3.1 emits an
independent binary objectness logit for every query, whereas the pilot had
treated query indices as mutually exclusive classes with softmax cross-entropy.
The next controlled run keeps matched-query mask supervision and score-based
inference, but uses SAM3.1-aligned binary objectness targets: the matched query
is positive and all remaining queries are negatives. The legacy softmax loss
is retained as an explicit ablation.

## Binary-objectness replication

Correcting the score loss improved some individual runs but did not remove
seed sensitivity. With loss weight 1, FTG versus Frame Prompt was 30.94 versus
27.29 overall at seed 11, but 21.47 versus 27.91 at seed 23. With weight 5, the
contrast widened in opposite directions: 34.36 versus 18.10 at seed 11 and
24.94 versus 34.16 at seed 23. Selection accuracy remained between roughly one
and eight percent. The larger weight is therefore rejected rather than chosen
from its favorable seed-11 result.

The factorization controls also weaken a direct FTG attribution. At seed 11,
Identity Only reached 30.96 overall, statistically indistinguishable at this
pilot scale from FTG's 30.94. Under an eight-frame budget, FTG reached 24.86
versus 28.34 for Frame Prompt. These outcomes do not support a robust gain from
the gated identity/state residual itself.

Frame-level Hungarian assignment provides a likely mechanism for the remaining
instability: the matched positive object-query slot can change independently at
every frame, even though FTG's identity representation is persistent. The next
test therefore uses one training-time query assignment for the entire video,
chosen by mean mask loss over frames. Inference remains the frozen SAM3.1 score
argmax; no candidate matcher, verifier, or ground-truth inference selection is
added.

## Query-slot consistency tests

The frame-wise FTG checkpoint changed its training-time matched query on 80.56%
of adjacent frame pairs, while the query selected by SAM3.1's score changed on
55.00%. This directly quantifies the slot permutation visible as identity drift
in crowded-video qualitative results.

Hard consistency did not solve the task. With one video-level training
assignment, FTG reached 28.77 versus 31.82 for Frame Prompt at seed 11 and 24.02
versus 28.80 at seed 23. Under the eight-frame budget it reached 29.15 versus
29.67. Selecting one inference query from the clip-averaged SAM score forced the
predicted switch rate to zero but further reduced seed-11 FTG from 28.77 to
26.06. State Only with video-level matching reached 23.99. A decoder query is
therefore not a reliable persistent identity carrier: unconstrained selection
switches identities, while fixing a slot suppresses legitimate per-frame mask
adaptation.

The next interface test removes the large, permutation-symmetric detector query
bank. It injects each learned prompt into one official SAM3.1 tracker object
slot and retains only the tracker's small multimask ambiguity set. This path is
an architectural grounding-interface control, not post-hoc matching, and uses
about half the pilot memory of the detector path.

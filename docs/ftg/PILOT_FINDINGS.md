# FTG pilot findings log

## Checkpoint-conversion correction (2026-10-04)

The initial SAM3 training-format checkpoint retained the HF-only `g_weight`
names for the two CXBlock layer-scale tensors under
`maskmem_backbone.fuser.layers.{0,1}`, while the training model expects
`gamma`. Because foundation loading used `strict=False`, both keys were silently
ignored and the two video-memory tensors remained randomly initialized. All
training-format SAM3 results recorded below (broad FTG, Frame Prompt, Anchored
FTG, and the unconditioned residual) are therefore **invalid until re-exported
with the corrected foundation weights**. They must not be used in paper tables
or method decisions. The public HF foundation baseline is unaffected.

The loader now conditionally maps a source `g_weight` key to `gamma` only when
that destination exists. A tensor-level audit after repair found zero differences
across all 1,459 public-foundation tensors after the normal dtype cast. As an
end-to-end check, the repaired Anchored export forced to `identity_memory`
exactly reproduced the public checkpoint on the same evenly spaced 128-expression
MeViS-v2 subset: both obtained J&F=64.813826%. Re-evaluation of the learned
Anchored FTG interface on those same keys obtained J&F=64.634527%, a controlled
change of -0.179299 points rather than the previously reported -16.12 points.
On the complete 907-expression split, repaired Anchored FTG obtained J=57.80,
F=66.29, and J&F=62.04 versus the public foundation's 57.69/66.22/61.95. The
overall change is +0.09 points. Target-present J&F was effectively unchanged
(64.589 versus 64.599); the small overall difference comes from no-target
expressions and is not evidence for identity-state factorization. The result
does establish that the corrected zero-initialized interface preserves the
strong foundation. Broad FTG and Frame Prompt must now be re-exported before
their controlled comparison can be interpreted.

The repaired broad checkpoint was next evaluated on the same deterministic
128-expression subset. Its two-token FTG interface reached J&F=64.704607%,
0.109219 points below the public foundation. Switching that *same checkpoint*
to a single `identity_memory` token reached 65.371112%, or +0.557286 points over
the foundation and +0.666506 points over two-token FTG. Target-present J&F shows
the same contrast (68.219 versus 67.519 versus 67.634 for broad Identity, broad
FTG, and the public foundation). Thus public-video Qwen/identity adaptation is
promising, but the current extra state token removes its gain. A complete
identity-adapted evaluation reached J=58.80, F=67.31, and J&F=63.06 on all 907
expressions, +1.10 points over the public foundation. Target-present J&F also
improved from 64.60 to 65.38 (+0.78), so the gain is not solely an absent-target
artifact. The next method gate should preserve this adapted identity base and
learn state only as a zero-initialized second stage, rather than jointly moving
Qwen and introducing a new prompt interface. Because this checkpoint was still
trained through the joint FTG interface, a clean single-prompt Identity
Adaptation run with the same data and budget is required for final attribution.

The repaired Frame Prompt checkpoint fails the same 128-expression controlled
gate: J=19.38, F=23.20, and J&F=21.29, with target-present J&F=19.24. This is
43.52 points below the public foundation and 44.08 points below the repaired
single Identity interface on identical evenly spaced keys. Frame-specific
prompts alone therefore do not explain the Identity gain and are not advanced
to full-split evaluation. Their severe regression is consistent with
frame-local referent decisions losing the stable cross-frame identity prior.

Qualitative extremes confirm that identity adaptation changes instance
selection rather than merely mask boundaries. It corrects near-total target
swaps for “the advancing cow that was the first to come closer” (present-frame
J 0.003 to 0.864) and the elephant being attacked by the others (0.009 to
0.877). Conversely, it changes a correctly tracked descending turtle to a
different turtle (0.881 to 0.003) and switches the referent of “the girl is
circling around what?” from the bicycle to the girl (0.720 to 0.025). The eight
ranked temporal sheets and manifest are stored in the 128-expression result's
`qualitatives/` directory. This bidirectional re-ranking is why a second-stage
state branch must retain an explicit fixed identity base.

This file records pilot outcomes, including negative evidence. Values are held-out
J&F percentages on a 48-expression development pilot (36 train, 12 validation),
not benchmark results.

## Strong public foundation baseline

The public `Sa2VA-Qwen3-VL-4B-SAM3` checkpoint was reproduced on the complete
local MeViS-v2 `val^u` split before FTG training. Native-resolution evaluation
over all 907 expressions, including absent-object frames with the official
DAVIS/Long-RVOS empty-mask convention, produced J=57.69, F=66.22, and
J&F=61.95. The raw prediction file and metric summary are stored under
`evo_artifacts/results/ftg/20261004_sa2va_qwen3_sam3_mevisu_baseline/MEVIS_U/`.
On the 869 target-present expressions alone it reached J&F=64.60; the official
overall score also contains 38 native MeViS-v2 no-target expressions.

An earlier local diagnostic reported J=59.37, F=53.25, and J&F=56.31 over only
869 expressions. It skipped empty-target frames and used normalized mean
boundary distance rather than the official DAVIS boundary tolerance. That file
is retained as `metrics_distance_legacy.json` for provenance and must not be
used in a paper table.

This is 18.50 J&F points above the 43.45 three-seed mean of the scratch Qwen
Frame Prompt pilot. The strong public checkpoint is therefore the primary
foundation; scratch Qwen remains a foundation ablation. This comparison changes
the initialization and is not evidence for FTG itself.

## Invalidated: initial full public-data FTG export

The first strong-foundation FTG run trained for one epoch on the complete public
MeViS-v2 and Long-RVOS train splits (3,517 samples, 440 distributed optimizer
steps). SAM3 was frozen; Qwen LoRA, the existing text projection, and the new
identity-conditioned state module were optimized. It completed normally on
eight A800 GPUs without NaNs or OOMs.

On the complete MeViS-v2 `val^u` split, official native-resolution scoring gave
J=43.87, F=51.00, and J&F=47.43. This is **14.52 J&F points below** the frozen
strong checkpoint. The result rejects this training recipe as a paper method.
The same conclusion holds after removing the 38 no-target expressions: FTG
reached 49.31 versus 64.60 for the foundation, a 15.29-point regression.
It does not yet distinguish two coupled changes: the new second sparse state
token and broad adaptation of 1.041B Qwen/embedding parameters on only 3,517
video samples. A same-budget Frame Prompt run is the immediate controlled test.
If broad adaptation is the shared failure, the next FTG variant must preserve
the strong checkpoint exactly at initialization: freeze Qwen and the pretrained
identity projection, and learn only a zero-initialized dynamic residual.

Full-split qualitative ranking confirms that this is an identity-grounding
failure rather than ordinary boundary noise. On “the individual giving
sustenance to the lizard,” the foundation tracks the referred arm/person while
FTG becomes nearly empty (J 0.904 to 0.019). On “cow walking to the front
first,” FTG consistently switches from the correct smaller cow to the adjacent
large cow (J 0.867 to 0.039). Conversely, FTG corrects whole-clip identity
mistakes on an airplane and a two-horse clip by more than 0.90 J. The broad
adaptation is therefore remapping which instance the language denotes, not
learning a controlled frame-state correction. Reproducible four-frame sheets
and their manifest are stored beside the FTG metric file under `qualitatives/`.

## Invalidated: initial frame-prompt and foundation-preserving exports

The same-budget Frame Prompt control reached J=22.85, F=26.84, and J&F=24.84
on all 907 MeViS-v2 `val^u` expressions. Its target-present J&F was 24.24. The
late training loss had already shown persistent divergence, and the complete
official score confirms that replacing the pretrained prompt with an
independently learned frame state is not a viable explanation or method.

Anchored FTG then froze Qwen, the pretrained `[SEG]` projection, and SAM3 and
trained only 592,897 parameters in a zero-initialized identity-conditioned
state residual. This version was exactly the public foundation at step zero and
trained without persistent divergence. Nevertheless, it reached only J=43.86,
F=50.94, and J&F=47.40; target-present J&F was 49.17. It therefore remained
14.56 points below the 61.95 foundation overall and 15.43 points below it on
target-present expressions. Foundation freezing and zero initialization alone
do not protect the pretrained language-to-instance mapping.

The qualitative failure is again instance selection rather than boundary
quality. On “cow walking to the front first,” Anchored FTG switches from the
correct smaller cow to the adjacent foreground cow (present-frame J 0.867 to
0.035). Three paraphrases of a stationary white car fall from about 0.915 to
0.088. Conversely, four paraphrases of a right-moving airplane improve from
0.002 to 0.908 by correcting the foundation's instance choice. Thus even the
small learned prompt displacement globally re-ranks instances in both
directions instead of acting as a local state correction. Reproducible sheets
and their manifest are stored under the Anchored result directory's
`qualitatives/` folder.

The next controls separate two questions. An unconditioned residual keeps the
same frozen foundation, zero initialization, trainable parameter count, data,
and schedule while removing identity from state extraction. A bounded FTG
variant additionally constrains residual norm to a fixed fraction of identity
norm, turning preservation into a functional prompt-space trust region rather
than inferring it from small weight norms. Neither control is promoted to the
paper method until it passes the same full-split gate.

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

## Stable tracker-slot interface

The tracker-slot control removed most query permutation but also removed the
useful segmentation candidates. Frame Prompt reached only 15.11 overall J&F at
seed 11, with a matched-query upper bound of 15.78; FTG reached 15.07. At seed
23, Frame Prompt and FTG reached 15.10 and 15.70. All six seed-11 variants fell
in the narrow 15.06--15.50 range. The issue is therefore decoder capacity for
this prompt type, not query selection. A raw projected language vector is not a
valid substitute for the visual object embedding expected by the tracker mask
decoder.

Inspection of the official SAM3.1 detector showed that its score head compares
object queries against the complete native language-token sequence. The next
interface preserves that pretrained native text grounding as a semantic
identity anchor and introduces Qwen Frame/FTG prompts only as a near-zero visual
residual. This makes the initial model equivalent up to a 1e-3 residual scale to
the public SAM3.1 text baseline, instead of asking a small pilot to relearn the
entire Qwen-to-SAM semantic space.

## Native-text anchor and temporal-budget replication

The native SAM3.1 text anchor substantially improved the controlled baseline and
produced the first cross-seed repeatable FTG contrast at an eight-frame budget.
At seed 11, FTG reached 38.99 overall J&F versus 36.03 for Frame Prompt; at seed
23 it reached 39.01 versus 36.04. The overall gains were therefore +2.96 and
+2.98 points. Long-RVOS Hybrid also repeated closely (+5.63 and +5.67), and
MeViS motion improved by +4.05 and +2.02. Long-RVOS Dynamic was unchanged
(+0.00 and -0.02), while Static changed by -0.01 and +6.15.

This is encouraging but not yet evidence of a selectively dynamic gain. Each
Long-RVOS type has only two validation expressions, Dynamic remains near zero in
absolute J&F, and the seed-23 Static gain is larger than the mean motion-sensitive
gain. The result justifies a larger replication; it does not by itself satisfy
the stronger identity/state mechanism claim.

## Dynamic-only residual and query-ranking failure

A stricter factorized interface treated frozen SAM3.1 native text as the identity
base and let FTG contribute only its gated dynamic-state residual. Without an
identity-aware association score, this version failed consistently. At 16 frames
with binary objectness, FTG versus Frame Prompt was 25.46 versus 34.36 at seed 11
and 27.01 versus 30.32 at seed 23. Replacing binary objectness with softmax
cross-entropy still gave 34.62 versus 38.62 at seed 11. The eight-frame contrast
was also inconsistent (-0.76 at seed 11, +1.28 at seed 23).

The negative actual scores coexist with very strong candidate-mask upper bounds:
FTG oracle J&F reached 85.98 at seed 11 and 81.57 at seed 23, and reached 90.41
under the softmax ablation. Qualitative examples likewise show good masks in the
candidate set but switches between people and temporal false-positive residue.
Thus factorizing only the prompt content is insufficient: persistent identity
must also participate in candidate association.

The next interface uses frozen SAM object-query features already produced by the
official decoder. A shared end-to-end alignment head scores those candidates
against a persistent identity representation for FTG and against a frame-varying
monolithic representation for Frame Prompt. The score is trained with the same
one-target supervision and used identically at inference; it is part of the
grounding interface, not an oracle, verifier, or post-hoc matcher.

## Identity-aware query association

The first association head used a learned cosine term with initial scale 1.
It failed consistently: FTG minus Frame Prompt was -6.50, -2.70, and -8.19
overall J&F at seeds 11, 23, and 31 under the eight-frame budget. At 16 frames,
the seed-11 difference was -4.44. FTG's oracle J&F nevertheless remained between
82.35 and 89.47 for the two primary seeds. Persistent identity alone therefore
does not calibrate the frozen decoder's candidate scores in this small training
regime.

Score-magnitude logging explained one concrete issue. Native SAM query logits
had standard deviation 1.0--1.24, whereas cosine alignment had standard
deviation only 0.05--0.07; the learned scale stayed near 1.005 after 72 updates.
A scale-10 softmax-ranking replication made the terms comparable and improved
Long-RVOS Hybrid by +7.15 and +9.58 points at seeds 11 and 23. However, overall
J&F still fell by 4.09 and 1.42 points because Static, Dynamic, and MeViS did
not retain their performance. This stronger ranker is also rejected rather than
selected for its favorable Hybrid subset.

The next scale gate returns to the only cross-seed-positive configuration: the
native-text anchor with a small full FTG residual at T=8. It expands to 192
public expressions (144 train, 48 video-disjoint validation), four epochs, three
optimization seeds, and Identity/State controls. This determines whether the
repeatable +2.96/+2.98 small-pilot gain survives adequate coverage.

## Scaled native-anchor T=8 replication

The 192-expression replication decisively rejected the small-pilot result. Over
three optimization seeds, Frame Prompt reached 43.45 +/- 0.31 overall J&F while
FTG reached 39.62 +/- 2.25, a delta of -3.83 +/- 1.96 points. The individual
overall deltas were -1.07, -4.99, and -5.44; none was positive. Mean FTG-minus-
Frame differences were -10.12 Static, -0.76 Dynamic, -6.73 Hybrid, and -1.80
MeViS points. The preregistered scale decision is therefore **NO-GO**.

The seed-11 controls further localize the failure. Identity Only reached 43.40
overall and State Only reached 43.51, both close to Frame Prompt at 43.84 and
above FTG at 42.77. Adding the two representations through the current gated
prompt composition is harmful; neither data coverage nor one isolated factor is
the primary problem. The small 12-example validation gain was driven by a few
large per-example corrections and did not generalize to 48 validation examples.

This closes the vector-composition design. The next architectural test must bind
persistent identity to an actual tracked object rather than to another detector
prompt vector: initialize an official SAM3.1 video tracker from a native-text
detector output, preserve identity in tracker memory, and reserve dynamic state
for frame-dependent updates. A raw language projection into a tracker slot and
post-hoc candidate matching remain excluded by the prior negative controls.

## Official detector-to-tracker foundation audit

The frozen official SAM3.1 video path was evaluated on the same 48-expression,
eight-frame held-out partition used by the scaled replication. Text detection
initialized the official tracker, and one tracker identity was selected once by
the model's native score and retained across the clip. Ground truth was not used
for initialization, selection, or propagation.

This persistent-track baseline reached 33.66 overall J&F: 65.20 Static, 25.29
Dynamic, 24.42 Hybrid, and 29.02 MeViS motion. It failed to expose any valid
track on 25 of 48 expressions (4/8 Static, 6/8 Dynamic, 5/8 Hybrid, and 10/24
MeViS), yielding a 70.55% false-reject rate. Unioning all official tracks was
retained only as a diagnostic and reached 36.43 overall; it is not the
single-referent primary output.

The same-manifest Frame Prompt mean was 43.45 J&F. The native tracker is much
stronger on Static (65.20 versus 51.42) but substantially weaker on Dynamic,
Hybrid, and MeViS. Persistent memory is therefore useful once identity is
acquired, but native text detection often never acquires the dynamically
described referent. This rejects using unmodified SAM3.1 tracking as the final
method while motivating the next recurrent interface more sharply: Qwen
frame-state grounding supplies observations and learned memory updates, whereas
the official tracker state carries persistent identity. Identity and state are
no longer composed into one detector-prompt vector.

Result artifact:
`/9950backfile/chenjiahui/evo_artifacts/results/ftg/20261004_sam31_video_baseline_scaled192_t8/result.json`.

## Qwen-mask initialization of tracker memory

The next training-free control used the trained Frame Prompt model to select an
anchor frame by native SAM query score, wrote its predicted full-resolution mask
into the official SAM3.1 multiplex memory encoder, and propagated that identity
in both temporal directions. Ground truth was used only for metrics. This is a
stronger and more faithful test than the earlier point-prompt control: the
tracker receives the complete Qwen-predicted mask through its native
`add_new_masks(..., add_mask_to_memory=True)` path.

Across all 48 held-out expressions, Frame Prompt reached 43.65 J&F in this
checkpoint evaluation, whereas Qwen-seeded tracking reached only 21.25 J&F.
Present-frame J&F fell from 51.20 to 9.89 and false rejection rose from 0.00% to
84.78%. The breakdown was 28.71 Static, 13.88 Dynamic, 12.58 Hybrid, and 24.10
MeViS motion J&F. The failure is therefore systematic rather than a single
endpoint-cache or point-prompt artifact.

This result rejects a training-free "predict one anchor mask, then propagate"
implementation of FTG. The persistent state must be learned jointly with the
frame observations, or integrated into a segmentation model whose video memory
was already trained for language-conditioned RVOS. The 144-example scratch-Qwen
pilot remains a controlled diagnostic, not the foundation for the full paper;
the next full-scale implementation should start from a public trained RVOS
checkpoint and retain scratch Qwen as an ablation.

Result artifact:
`/9950backfile/chenjiahui/evo_artifacts/results/ftg/20261004_qwen_seeded_tracker_scaled192_t8_v3/merged/result.json`.

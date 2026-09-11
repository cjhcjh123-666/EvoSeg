# M5 pilot report — referent commitment and evidence volume

**Issue**: [#3 — M5 Research: Premature Referential Commitment (falsification-first pilot)](https://github.com/cjhcjh123-666/EvoSeg/issues/3)
**Branch**: `research/premature-commitment-pilot`
**Base commit**: `730451a7d07c692c1391b1d83c0b68ca2059cdf7` (main at task creation)
**Model audited**: `EvoSeg-Qwen3-VL-4B-Faithful` (image-Faithful base + external v6 temporal head)
**Data**: Ref-YT-VOS `valid` (annotation dirs are keyed by *expression id*) plus the
project's own `faithfulness_valid.json` manifest for case selection.

> **Revision 2 (this file).** Review of revision 1 found two evaluation defects and
> one over-stated claim. All three are fixed, the affected numbers are recomputed,
> and §7 changes the verdict wording accordingly. Summary of the changes:
>
> | defect | effect on revision 1 | fix |
> |---|---|---|
> | identity was measured on `prediction_masks` (= raw \* e_t), i.e. **after** the temporal verifier gate, because the audit loads the v6 head | conflated "wrong instance" with "verifier abstained"; an empty frame scores `id_err = 0` | **P0-A**: identity is now measured on `raw_masks` (pre-gate SAM2 propagation); the gated stream is recorded and reported separately (§3.4) |
> | prefix comparisons used each prefix's **own** horizon (0.2 ≈ 5 frames, 1.0 ≈ 24 frames) | a shorter horizon is easier, so short prefixes were flattered; the "identical first-5-frames" pairing was not actually identical (80 vs 100 frames) | **P0-B**: every cross-prefix number now uses a strict per-case window `W_i = min(sam2_prompt_frames)`, the same frames `0..W_i-1` for both sides (§3.2) |
> | `cos = 0.96` was read as "the referent representation is not revised" with no scale | the claim had no calibration | extra run per case with a **different query** on the same video gives the reference scale (§3.3) |

This pilot is **falsification-first**: its job is to decide whether the hypothesis
survives, not to produce a module. No Agent, RL, verifier, belief network or
SAM3.1 integration was built — that is the M5 stop condition, and it was respected.

---

## 0. Document map

| milestone | commit | deliverable |
|---|---|---|
| M5.0 audit | `8feecce` | `docs/premature_commitment/P0_inference_audit.md` — inference information-flow audit (code + measurement) |
| M5.1 data | `abea063` | `docs/premature_commitment/P1_candidate_mining.md` — ambiguous-referent candidate pool |
| M5.2 annotation | `d16a0af` | `docs/premature_commitment/P2_pilot_protocol.md` — annotation bundle + protocol (**unlabelled**) |
| M5.3 eval | `4559c33` | prefix identity / headroom diagnostics (revision 1, superseded) |
| M5.4 report | `d167139`, `b02664a` | revision-1 report |
| M5.5 corrections | *this revision* | raw-mask audit (schema v2), strict common window, cosine calibration, controlled VLM-evidence A/B, gate analysis |

Committed results (revision 2):

```
docs/premature_commitment/results/P0_prefix_audit_v2_identity_swap_n20.json          (+ .meta.json)
docs/premature_commitment/results/prefix_identity_eval_v2.json                       (+ .meta.json)
docs/premature_commitment/results/delayed_oracle_headroom_v2.json                    (+ .meta.json)
docs/premature_commitment/results/reproduction_check.json
docs/premature_commitment/results/candidate_pool.json / candidate_summary.csv / candidate_stats.json
docs/premature_commitment/results/annotation_bundle_index.json
docs/premature_commitment/figures/prefix_identity_curve_v2.png
```

Revision-1 artifacts stay in the tree for traceability and are labelled as
superseded. Heavy artifacts (per-frame masks, the 4 817-frame annotation image
set, shard JSONs) live under `<artifact_root>/results/premature_commitment/` and
are not committed.

---

## 1. What the hypothesis was, and what the pilot may conclude

> Modern referring-video segmentation may suffer from **hard referential
> commitment**: it collapses temporally evolving and potentially ambiguous
> evidence into a single referent hypothesis, which can cause persistent
> identity errors.

The issue pre-registered three candidate mechanisms, to be decided from code *and*
measurement:

* **A — prefix-causal premature commitment**: the model commits before the
  evidence needed to disambiguate has arrived, and more evidence would fix it.
* **B — offline hard-referential commitment / single-hypothesis bottleneck**: the
  model has all the evidence, but the interface forces one hypothesis and offers
  no way to revise it.
* **C — neither**.

## 2. What was run

1. **P0 inference audit** (v2, schema v2). Same 20 `identity_swap` cases (selected
   from the manifest's own GT-derived category, so *not* selected for baseline
   failure), each run under VLM evidence 20 / 40 / 60 / 80 / 100 % of the video
   (`vlm_all_frames=True`), plus the default path (`vlm_all_frames=False`: only
   the first 5 frames reach the VLM) on the full video, plus **one extra run with
   a different expression of the same video** for the cosine calibration. Every
   run records both mask streams: `raw_masks` (SAM2 propagation before the
   verifier gate — **primary**) and `prediction_masks` (= raw \* e_t — secondary),
   per-frame target IoU, max-distractor IoU, `identity_margin`, `IDErr`, empty
   flag, the `[SEG]` vector, and the prompt/propagation frame counts implied by
   the architecture. 140 model runs; the raw stream was available for 20/20 cases
   (no fallbacks).
2. **P1 candidate pool**: 834 real Ref-YT-VOS expressions mined → pool 250 →
   stratified top-80 (see `P1_candidate_mining.md`).
3. **P2 annotation bundle**: 80 cases × 806 checkpoints, 3 annotators, **no
   labels exist yet** (see `P2_pilot_protocol.md`).
4. **Diagnostics**: common-window prefix curve, paired tests, the controlled
   VLM-evidence A/B, gate analysis, cosine calibration and `ORACLE_*` headroom.

## 3. Results

### 3.1 Controlled A/B: VLM sees the first 5 frames vs the whole video

This is the cleanest experiment in the pilot, and it has **no horizon confound at
all**: the default path and the 100 %-prefix run have the same SAM2 prompt frames
(`min(5, T)`), the same propagation horizon (the whole video) and the same query.
The *only* difference is whether the VLM's single forward pass contained every
sampled frame or only the first five. Both are evaluated on frames `0..4` of each
case (5 frames × 20 cases = 100 frames per side).

| on the identical frames | VLM sees first 5 | VLM sees all frames | paired delta |
|---|---|---|---|
| `IDErr` (case-macro) | **25.0 %** [8.0, 45.0] | **35.0 %** [16.0, 55.0] | **+10.0 pp [+1.0, +22.0]** |
| identity margin | **+0.299** [+0.027, +0.541] | **+0.195** [−0.080, +0.456] | −0.104 [−0.283, +0.018] |
| `IoU_target` (frame-micro) | 0.482 | 0.432 | — |
| `IoU_distractor` (frame-micro) | 0.183 | 0.237 | — |
| `IDErr` (frame-micro, 100 vs 100 frames) | 25.0 % | 35.0 % | — |

The aggregate identity decision flips in 1/20 cases, but the frame-level error
rate rises significantly. **Giving the VLM the entire video instead of the first
five frames makes its referent conditioning *less* discriminative on the very
same frames**, with the same segmentation and tracking behind it.

### 3.2 Prefix curve on a strict per-case common window (raw masks)

`W_i = min(sam2_prompt_frames)` over the compared prefixes, capped at 5; both
sides use exactly frames `0..W_i-1` (window: min 2, median 4, max 5; 80 frames per
prefix in total). Because `sam2_prompt_frames = min(5, n_used)`, every frame in
the window is prompted in every run, so within the window the runs differ only in
the language conditioning.

| VLM evidence | frames used | frames evaluated | **IDErr** (case-macro) | margin | IoU target | IoU distractor | IDErr (frame-micro) | decisions t/d/e |
|---|---|---|---|---|---|---|---|---|
| prefix 0.2 | 4.75 | 4.00 | **23.0 %** [5.0, 43.0] | **+0.308** | 0.493 | 0.185 | 23.8 % | 15/5/0 |
| prefix 0.4 | 9.50 | 4.00 | 31.0 % [15.0, 52.0] | +0.291 | 0.483 | 0.192 | 32.5 % | 14/6/0 |
| prefix 0.6 | 14.35 | 4.00 | 30.0 % [15.0, 50.0] | +0.275 | 0.478 | 0.203 | 31.2 % | 14/6/0 |
| prefix 0.8 | 19.15 | 4.00 | 31.2 % [13.8, 51.2] | +0.230 | 0.454 | 0.225 | 32.5 % | 13/7/0 |
| prefix 1.0 | 23.85 | 4.00 | **35.0 %** [15.0, 55.0] | **+0.195** | 0.434 | 0.240 | 36.2 % | 13/7/0 |

Paired prefix-0.2 vs prefix-1.0 on the identical window: `IDErr` **+12.0 pp
[0.0, +26.0]** (95 % bootstrap CI includes 0 — the per-case effect is not
significant at N = 20), margin **−0.114 [−0.316, +0.020]**; error transitions
`wrong_fixed 0 / wrong_persist 5 / right_right 13 / right_broken 2`; 2/20 decision
flips. **More evidence never fixed a wrong referent.**

The trend is monotone in the margin and in `IoU_distractor` (+0.185 → +0.240): as
the VLM sees more frames, the target and the distractor become harder to separate
on the frames that both runs can see.

**Confounded view, kept for continuity.** Averaging each prefix over its *own*
horizon gives `IDErr` 25.5 / 36.3 / 40.8 / 42.2 / 48.3 % for 0.2 → 1.0 with the
same monotone margin decay (+0.280 → +0.018). This is the number revision 1
quoted; it mixes the evidence effect with "later frames are where occlusion,
crossing and identity switches live", so it must not be used as the primary
comparison. It appears here only so the size of the confound (≈ +1.2 pp per
prefix step) is visible.

### 3.3 `[SEG]` similarity, now calibrated

Revision 1 reported `cos([SEG]@20 %, [SEG]@100 %) = 0.960` and concluded that the
referent representation "is formed once and not revised". A cosine needs a scale,
so each case now also gets a run with a **different expression of the same video**
(same frames, same default path), which measures how far the conditioning moves
when the *query* changes and nothing else does:

| perturbation | n | mean cos to the full-video, same-query vector | min | max |
|---|---|---|---|---|
| evidence volume (5 prefixes, all cases) | 100 | **0.988** | 0.787 | 1.000 |
| **different query, same video** | 20 | **0.761** (median 0.789) | 0.269 | 0.996 |

So changing the query moves the conditioning vector roughly 20× further than
growing the evidence 5×. **The statement "the referent conditioning is not
revised by additional evidence" survives, and is now calibrated.** Two caveats
must travel with it: (i) the vector is *also* 0.76-similar across genuinely
different queries, so it is not a sharply query-specific representation to begin
with — a 256-d vector that varies comparatively little with query content should
not be assumed to carry a resolved referent; (ii) high cosine does not by itself
prove the vector is unchanged in the dimensions that matter — the mask decisions
do move (§3.1, §3.2). The defensible phrasing is therefore: *highly correlated
across evidence volumes, and not systematically better for identity.*

### 3.4 Does the verifier gate change the identity numbers? (P0-A check)

Revision 1's numbers came from the gated stream; the corrected primary stream is
raw. Comparing both on the same 100 runs (20 cases × 5 prefixes, common window):

| quantity (case-macro over runs) | raw | gated |
|---|---|---|
| `IDErr` | 30.05 % | 28.80 % (paired −1.25 pp [−3.75, 0.0]) |
| empty-frame rate | 0.25 % | 2.50 % |
| identity-decision disagreements | — | **1 / 100** |

So on these 20 `identity_swap` cases the gate distorts the identity metric only
slightly — the correction is conceptually necessary (an abstained frame is not a
correct identity decision) but it is not what made revision 1's numbers
misleading. The horizon confound (§3.2) was the larger error, and the
`prediction_masks`/`raw_masks` issue would have become a serious distortion on
benchmarks where the verifier abstains often. Both are now fixed.

### 3.5 `ORACLE_*` headroom (demoted, horizon-matched)

| variant | IDErr | paired delta vs NORMAL |
|---|---|---|
| `NORMAL` (full video to the VLM, common window), raw | 35.0 % [15.0, 55.0] | — |
| `ORACLE_TIMING`, common window, raw | 23.0 % [5.0, 43.0] | −12.0 pp [−26.0, **0.0**] |
| `ORACLE_TIMING`, per-prefix horizon (confounded) | 25.5 % | −22.8 pp [−37.7, −10.3] |

`ORACLE_TIMING` picks the shortest prefix in 20/20 cases in both variants, so
≈11 pp of revision 1's −16.5 pp "headroom" was the horizon confound, not evidence
volume. What remains is consistent with §3.1/§3.2 (fewer VLM frames ⇒ better
target/distractor separation on the same frames), but it is an oracle over five
correlated options and its CI touches zero.

**This is therefore not used as evidence, not in the verdict, and not as a
headline.** It is reported to document how large the confound was.

`DELAYED` remains **not implementable**: one `[SEG]` → one language prompt on the
first `min(5, T)` frames → one causal propagation leaves no second commitment
point and no hypothesis state to delay. `ORACLE_ID` is **not implementable as a
method** for the same reason (there is exactly one hypothesis to choose from).
Per the issue's rules, both are recorded as such rather than hacked into a
variant that merely looks better.

### 3.6 Determinism and reproduction

* The revision-1 committed audit JSON was produced with the gated stream; re-running
  the v2 script and comparing the **gated** stream to it matches exactly:
  **720/720 discrete fields**, `max |Δ| = 0.0` on every continuous metric,
  100 % of frames identical (`reproduction_check.json`, run at the v2 commit SHA).
* All revision-2 numbers come from one 20-case run split over 4 GPUs (GPUs 0–3,
  one shard per GPU, `--shard-index/--shard-count`), and `raw_masks` was available
  in 20/20 cases with no fallback.

## 4. The six required questions, answered

**Q1 — Is ambiguity-before-identification real and common?**
Still **not answered**, because identifiability is exactly what P2 measures and no
human label exists. The miner establishes only that the structural preconditions
are common: in the stratified top-80, 78/80 cases have a same-category competitor
whose dilated mask overlaps the target, 58/80 have ≥2 same-category competitors,
31/80 contain target-absent annotated frames, 14/80 contain a
disappearance→reappearance gap, and the median number of jointly visible frames is
23.5 (pool of 250: 67 with absent frames, 30 with a gap).
**Assessment: structurally plausible and common; human-confirmed frequency
unknown.**

**Q2 — Does the model already make a single-object hard prediction under
ambiguous evidence?**
Behaviourally yes: with ~5 frames of evidence it already emits a non-empty single
mask in 19/20 cases with a committed instance (15 target / 5 distractor),
`IDErr` 23.0 % on the common window and 0 empty masks in the window. The stronger
claim — "it commits *before a human could decide*" — still needs the P2 labels and
is **not** established.

**Q3 — Does the problem persist once future evidence reaches the VLM?**
Yes, and this is now the best-controlled result in the pilot (§3.1): with identical
prompt frames and identical propagation, giving the VLM the whole video instead of
the first five frames **raises `IDErr` by +10.0 pp [1.0, 22.0]** and lowers the
identity margin by 0.104 on the same 100 frames. The failure is not "the model did
not see the future" — the future is in its single forward pass and does not help.

**Q4 — Does the delayed/oracle analysis show enough headroom?**
Not in a way that supports "delay commitment". The horizon-matched oracle gain is
−12.0 pp with a CI touching zero, and the oracle always selects the **shortest**
prefix; `DELAYED` cannot even be expressed in this architecture. The observable is
that *less* VLM evidence yields a more discriminative conditioning on the same
frames — a statement about how the token aggregates evidence, **not** about
timing. Nothing here licenses a "wait longer" method.

**Q5 — Which formulation does the evidence support?**
The **mechanism in code** is B-shaped (§1): exactly one `[SEG]` per query, one
prompt set on the first `min(5, T)` frames, one causal propagation, no state for a
competing hypothesis, and a conditioning vector that is nearly invariant to
evidence volume (calibrated: 0.988 vs 0.761 for a query change).
**But B must be reported as a *candidate* finding, not as the established root
cause of the identity errors.** The pilot measured (i) the single-hypothesis
interface, (ii) evidence-insensitivity of the conditioning, and (iii) that
identity does not improve — and in the controlled A/B *degrades* — when the VLM
sees more. None of those three is a causal experiment on hypothesis multiplicity:
that would need an architecture that can hold several hypotheses, or an
`ORACLE_ID`–style diagnostic that localises the error to the VLM→prompt interface
versus SAM2 propagation. Neither exists in M5.

**Q6 — GO or NO-GO?** → §7.

## 5. What is missing, and what it would take to close it

* **Human identifiability labels — still absent, still required.** `t*` must not be
  generated by a model, and the pilot had no annotator budget. Bundle and tool are
  ready (P2): 80 cases, 806 checkpoints, 3 annotators, ≈2 418 judgements (≈1.8 h
  per annotator; a defensible reduction is first/middle/last checkpoint per case =
  240 per annotator, keeping ≥60 cases). Entry point:
  `<artifact_root>/results/premature_commitment/annotation_bundle/annotate.html`.
  Until then these stay **N/A**: pre-identifiability hard-prediction rate,
  pre-identifiability wrong-identity rate, post-identifiability identity error,
  commitment lag `Δt = t − t_identifiable` (helpers exist in `metrics.py`,
  deliberately unused). Pre-registered abort rule: label 30 sampled checkpoints
  first and stop if >40 % are `INVALID`/`UNCERTAIN` for all annotators.
* **N = 20, one model, one category.** The diagnostics are a feasibility probe on
  the 4B Faithful model on `identity_swap`; CIs are wide (±10 pp at best) and the
  horizon-matched window is shallow (median 4 frames), because the shortest prefix
  bounds every comparison.
* **No multi-hypothesis architecture to test causality of B.** Until a model that
  emits several `[SEG]` tokens (or an isolation diagnostic) is available, "single
  hypothesis" stays a candidate explanation.
* **No frame-level occlusion/disappearance GT** beyond the dataset's own
  every-5th-frame annotations; "visible" is derived from GT mask non-emptiness.
* **The cosine calibration uses the first other expression** of the same video,
  which may itself be unusually similar or dissimilar — 20 samples, median 0.789.
* **The `Δz → Δmask` sensitivity was not measured.** The cosine calibration fixes
  the *scale* of the conditioning drift, but it does not say how the drift maps
  onto mask decisions (e.g. per case, does a larger `Δz` predict a larger
  `Δmargin`/`IDErr`?). That is a cheap follow-up on the existing JSONs, and it
  would strengthen or kill the "the token is not being used to revise the referent"
  reading; it is not in this revision.

## 6. Threats to validity / honest caveats

1. The 20 cases come from the project's own GT-derived `identity_swap` bucket: a
   legitimate, failure-independent construction, but small and constructed.
2. The paired same-window argument relies on SAM2 propagation being causal
   (verified in code: one `propagate_in_video` pass, prompt applied only on the
   first `min(5, T)` frames). §3.1 does **not** rely on it: there the two runs have
   the same prompt frames *and* the same propagation horizon.
3. The horizon-matched window is short (median 4 frames) because the shortest
   prefix defines it. Conclusions about "evidence volume" therefore apply to the
   early part of each video; the full-horizon curve (§3.2, confounded) is the only
   description of the pipeline's behaviour later in the video.
4. The 95 % CIs for the paired deltas are wide and two of them include zero
   (`+12.0 pp [0, +26]` for 0.2→1.0; `−12.0 pp [−26, 0]` for the oracle). Only the
   controlled A/B in §3.1 excludes zero. Small-sample honesty: the *direction* is
   consistent across every analysis, the *magnitude* is not yet tight.
5. Two defects in revision 1 were found by review and are fixed here; the fix to
   the run metadata (git SHA, per-mode `vlm_all_frames`) from the previous revision
   still applies, and `candidate_stats.json` now labels its scopes
   (`mined_*`/`pool_*`/`topk_*`).

## 7. GO / NO-GO Decision

> **NO-GO for the old hypothesis.**
> **That is not a NO-GO for EvoSeg, and not a NO-GO for the problem of referent
> identity in video.** The pilot removed a wrong mechanistic explanation.

**1. The old hypothesis is dead: NO-GO.**
"The model commits before the evidence arrives, and more evidence would fix it" is
rejected, and rejected for structural reasons, not for lack of data: under the
official runner the VLM already consumes the whole sampled video in one forward
pass; the `[SEG]` conditioning moves less when the evidence grows 5× (0.988) than
when the query changes (0.761); zero of five wrong commitments were corrected by
4× more evidence; and in the controlled A/B, *adding* the full video to the VLM
makes identity worse (+10.0 pp `IDErr` [+1.0, +22.0], identical frames, identical
propagation). The word *premature* must not be used as a conclusion.

**2. What survives is a sharper, better-measured question — not a method mandate.**
The claim that a **single query-level segmentation token is insufficient to resolve
a temporally defined referent** is now the best-supported formulation, and it is
supported as a *candidate finding* (formulation B), not as a proven root cause:
the pilot measured the single-hypothesis interface, the evidence-insensitivity of
the conditioning, and the failure of identity to improve, but it did not run the
causal experiment (multiple hypotheses, or an `ORACLE_ID` isolation diagnostic).
Concretely, the defensible summary is:

> The model already sees the whole video, and the whole video is not being turned
> into a more reliable referent: on identical frames and identical propagation,
> more VLM evidence raises identity error and shrinks the target/distractor margin.

**3. Under this verdict, no method work is authorised now** — no belief model, no
SAM3.1 integration, no Agent, no RL, no new verifier (M5 stop condition).

**4. Pre-registered next step (only after these two fixes, and only if the
reviewer wants it).** The pilot's open question is *granularity*, not timing: the
query describes a **trajectory** $\mathcal{T}_i = (M^i_1,\dots,M^i_T)$, but the
pipeline compresses the video into one query-level token that conditions a mask
prompt. The cheap falsification test is a **trajectory oracle**:

```text
given GT candidate trajectories {T_1 ... T_K} (GT masks, no segmentation involved)
and the query q, select k -- can language pick the right trajectory?
```

Feasibility gate, stated before running it: if trajectory-level grounding with GT
candidates drives identity error from ≈30–40 % down to **<10–15 %**, the
"language refers to trajectories, not propagated masks" story earns a method
paper. If it does not, the bottleneck is elsewhere (e.g. the language→visual
grounding itself), and the project should not be built on the trajectory framing.
This is a *diagnostic*, not a method, and it is not part of M5.

## 8. Reproduction

```bash
export PYTHONSAFEPATH=1
A=/9950backfile/chenjiahui/evo_artifacts/results/premature_commitment
P=projects/evoseg/premature_commitment

# M5.5 — raw-mask audit (schema v2), 20 cases split over 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python $P/audit_inference_information_flow.py \
    --n-cases 20 --category identity_swap --shard-index $i --shard-count 4 \
    --tag v2_shard$i --out-dir $A/v2
done

# diagnostics (CPU seconds; no model is loaded)
python $P/eval_prefix_identity.py --audit "$A/v2/*_shard[0-9].json" --out-dir $A/prefix_eval_v2
python $P/eval_delayed_commit.py  --audit "$A/v2/*_shard[0-9].json" --out-dir $A/headroom_v2

# determinism: the revision-1 file stored the GATED stream, so compare it to the
# gated block of the v2 runs
python $P/verify_reproduction.py --reference $A/P0_prefix_audit_identity_swap_n20.json \
  --reference-flat-stream gated --shard "$A/v2/*_shard[0-9].json" \
  --out $A/v2/v1_gated_vs_v2_gated.json

# candidate pool + annotation bundle (CPU; see P1/P2)
python $P/mine_ambiguous_candidates.py --out-dir $A --pool-size 250 --top-k 80
python $P/build_annotation_bundle.py --pool $A/candidate_pool.json \
  --out-dir $A/annotation_bundle --top-k 80 --max-checkpoints 12 --thumb-width 320

# smoke test (no model/dataset/GPU; standalone — this venv has no pytest)
python $P/tests/test_metrics_smoke.py
```

Every run writes a metadata JSON (git SHA, exact command, model/dataset/manifest
paths, seed, world size, per-mode `vlm_all_frames`, primary/secondary streams,
shard index). All paths are CLI-overridable. No `except Exception: pass` exists in
M5 code: degraded paths print and are recorded (`vlm_all_frames_fallback`,
`raw_unavailable_reason`, `matplotlib unavailable`).

## 9. Deliverable checklist (Issue #3)

| required | status |
|---|---|
| P0 inference audit; the 10 questions answered from code **and** measurement | ✅ `P0_inference_audit.md` (v2 revision included) |
| raw-mask identity (no verifier contamination) — added after review | ✅ P0-A, §3.4 |
| strict common-window prefix comparison — added after review | ✅ P0-B, §3.2 |
| calibrated `[SEG]` cosine (different-query scale) — added after review | ✅ §3.3 |
| prefix diagnostic 20/40/60/80/100 % on ~20 same-category multi-instance cases | ✅ 20 cases, 140 runs |
| 150–250 real candidate cases, no toy negatives, stable keys | ✅ 250 (top-80 stratified) |
| `candidate_pool.json` / `candidate_summary.csv` / statistics / HTML | ✅ (HTML in the artifact root) |
| human prefix-identifiability tool, ≥3 annotators, no model labels | ✅ `annotate.html`, 806 checkpoints |
| Experiment A (prefix identity curve), B (full-video vs prefix), C (delayed/`ORACLE_*`) | ✅ B strengthened into a controlled A/B |
| frame-micro + case-macro, N/mean/median/paired delta/bootstrap CI ≥1000, fixed seed | ✅ |
| reproducible scripts, CLI-overridable paths, run metadata JSON | ✅ |
| `pilot_report.md` with `## GO / NO-GO Decision` | ✅ (this file) |
| stop after the pilot; no belief model / SAM3.1 / Agent / RL | ✅ respected |

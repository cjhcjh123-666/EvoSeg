# M5 pilot report — Premature / hard referential commitment

**Issue**: [#3 — M5 Research: Premature Referential Commitment (falsification-first pilot)](https://github.com/cjhcjh123-666/EvoSeg/issues/3)
**Branch**: `research/premature-commitment-pilot`
**Base commit**: `730451a7d07c692c1391b1d83c0b68ca2059cdf7` (main at task creation)
**Model audited**: `EvoSeg-Qwen3-VL-4B-Faithful` (image-Faithful base + external v6 temporal head)
**Data**: Ref-YT-VOS `valid` (annotation dirs are keyed by *expression id*), plus the
project's own `faithfulness_valid.json` manifest for case selection.

This pilot is **falsification-first**. Its job is to decide whether the hypothesis
survives, not to produce a module. No Agent, RL, verifier, belief network or
SAM3.1 integration was built — that is the M5 stop condition and it was respected.

---

## 0. Document map

| milestone | commit | deliverable |
|---|---|---|
| M5.0 | `8feecce` | `docs/premature_commitment/P0_inference_audit.md` — inference information-flow audit (code + measurement) |
| M5.1 | `abea063` | `docs/premature_commitment/P1_candidate_mining.md` — ambiguous-referent candidate pool |
| M5.2 | `d16a0af` | `docs/premature_commitment/P2_pilot_protocol.md` — annotation bundle + protocol (**unlabelled**) |
| M5.3 | `4559c33` | `docs/premature_commitment/results/prefix_identity_eval.json`, `.../delayed_oracle_headroom.json`, `docs/premature_commitment/figures/prefix_identity_curve.png` |
| M5.4 | this file | pilot summary + GO / NO-GO, run-metadata repair, 4-GPU reproduction check |

Committed results:

```
docs/premature_commitment/results/P0_prefix_audit_identity_swap_n20.json   (+ .meta.json)
docs/premature_commitment/results/candidate_pool.json / candidate_summary.csv / candidate_stats.json
docs/premature_commitment/results/annotation_bundle_index.json
docs/premature_commitment/results/prefix_identity_eval.json               (+ .meta.json)
docs/premature_commitment/results/delayed_oracle_headroom.json            (+ .meta.json)
docs/premature_commitment/results/reproduction_check.json
docs/premature_commitment/figures/prefix_identity_curve.png
```

Heavy artifacts (per-frame masks, the 4 817-frame annotation image set) stay
under `<artifact_root>/results/premature_commitment/` and are not committed.

---

## 1. What the hypothesis was, and what the pilot was allowed to conclude

The hypothesis under test:

> Modern referring-video segmentation may suffer from **hard referential
> commitment**: it collapses temporally evolving and potentially ambiguous
> evidence into a single referent hypothesis, which can cause persistent
> identity errors.

The issue pre-registered that the *mechanism* could be one of three things, and
that the pilot must pick one **from code and measurement**, never from reading
alone:

* **A — prefix-causal premature commitment**: the model commits before the
  evidence needed to disambiguate has arrived, and would be fixed by evidence.
* **B — offline hard-referential commitment / single-hypothesis bottleneck**:
  the model has all the evidence, but the interface forces one referent
  hypothesis and provides no way to revise it.
* **C — neither**: the hypothesis about "hard commitment" is itself not
  supported (e.g. commitment is not hard, or the errors are segmentation
  errors rather than referent-selection errors).

## 2. What was actually run

1. **P0 — inference audit** (M5.0). Read `predict_forward`, the Ref-YT-VOS
   runner, the SAM2 wrapper and the identity code, answered all 10 questions
   with line references, then measured: for 20 same-category multi-instance
   (`identity_swap`) cases, the same video/query was re-inferred under VLM
   prefixes of 20 / 40 / 60 / 80 / 100 % (all frames inside the prefix reach
   the VLM) plus the default path (only the first 5 frames reach the VLM).
   Recorded: `[SEG]` presence, the `[SEG]`/segmentation-conditioning vector,
   cos(prefix `[SEG]`, full `[SEG]`), per-frame target IoU, max distractor IoU,
   `identity_margin = IoU_target − IoU_distractor_max`, per-frame `IDErr`,
   empty/non-empty, and the aggregate identity decision. 120 model runs total.
   Selection used the manifest's own `identity_swap` category, computed from GT
   instance tracks — not from baseline failures — so no failure-only bias.
2. **P1 — candidate pool** (M5.1). Mined 834 real Ref-YT-VOS expressions (53
   videos) for structural ambiguity signals; pool = 250; annotation subset =
   stratified top-80. No toy negatives of any kind.
3. **P2 — annotation bundle** (M5.2). 80 cases × 806 checkpoints, four-way
   label tool with 3 annotators. **No labels exist yet** — see §5.
4. **P3 — diagnostics** (M5.3). Prefix identity curve, full-vs-prefix paired
   comparison on *identical* frames, error-transition table, and
   delayed/ORACLE_* headroom analysis. All from the P0 JSON (no new inference).

## 3. Results

### 3.1 Prefix identity curve (20 `identity_swap` cases, case-macro)

| VLM evidence | mean frames | IoU target | IoU distractor | margin | **IDErr** | decisions (target/distractor/empty) | cos(`[SEG]`, full) | frame-micro IDErr |
|---|---|---|---|---|---|---|---|---|
| prefix 0.2 | 4.75 | 0.479 | 0.198 | **+0.280** | **25.5%** | 14 / 5 / 1 | 0.960 | 27.4% |
| prefix 0.4 | 9.50 | 0.442 | 0.252 | +0.190 | 34.5% | 13 / 7 / 0 | 0.986 | 39.5% |
| prefix 0.6 | 14.35 | 0.427 | 0.279 | +0.149 | 39.9% | 11 / 9 / 0 | 0.994 | 45.3% |
| prefix 0.8 | 19.15 | 0.404 | 0.295 | +0.109 | 40.4% | 12 / 8 / 0 | 0.998 | 44.4% |
| prefix 1.0 (full) | 23.85 | 0.362 | 0.315 | **+0.047** | **42.0%** | 10 / 10 / 0 | 1.000 | 45.1% |
| default path (VLM first-5) | 23.85 | 0.457 | 0.269 | +0.188 | 34.8% | 13 / 7 / 0 | — | 36.3% |

Bootstrap 95% CIs (1000 resamples, seed 0, case-macro): IDErr at prefix 1.0
**42.0% [23.8, 59.7]**, at prefix 0.2 **25.5% [9.7, 42.6]**. Identity margin at
prefix 1.0: **+0.047 [−0.198, +0.312]** — i.e. at full evidence the target and
the best distractor are statistically indistinguishable in mean IoU.

The curve is monotone in the *wrong* direction for hypothesis A: **more
evidence, worse identity**, both case-macro and frame-micro.

### 3.2 Paired comparison on identical frames

The prefix runs share the same SAM2 prompt frames (first `min(5, T)`), and SAM2
propagation is causal, so within the first 5 frames the two runs differ only in
the language conditioning produced by the VLM. Comparing prefix 0.2 against
prefix 1.0 **restricted to those identical first 5 frames**:

| quantity (first 5 frames) | low evidence (prefix 0.2) | full evidence (prefix 1.0) | paired delta |
|---|---|---|---|
| case-macro IDErr | 23.0% [6.0, 40.0] | 32.0% [12.0, 52.0] | **+9.0pp [1.0, 20.0]** |
| frame-micro IDErr (80 vs 100 frames) | 23.8% | 32.0% | — |
| case-macro margin | +0.262 | +0.189 | — |

Error transitions between low and full evidence: `wrong_fixed = 0`,
`wrong_persist = 5`, `right_right = 13`, `right_broken = 2`; the aggregate
identity decision flips in **5 / 20** cases.

**The model corrected 0 of 5 wrong-instance commitments when given 4× more
evidence.** And on frames it can see in both settings, more evidence made
identity *worse*, not better.

### 3.3 `[SEG]` is almost invariant to evidence volume

cos(`[SEG]`@20 %, `[SEG]`@100 %) = **0.960** mean (monotone to 0.986 / 0.994 /
0.998 at 40/60/80 %). The referent (conditioning) vector is formed once and is
essentially not revised as evidence accumulates — consistent with B, not A.

### 3.4 Delayed / ORACLE_* headroom

| variant | IDErr (case-macro) | note |
|---|---|---|
| `NORMAL` (official full-video path) | 42.0% [23.8, 59.7] | — |
| `ORACLE_TIMING` (prefix chosen per case with GT) | 25.5% [9.7, 42.6] | paired delta **−16.5pp [−28.3, −6.1]** — upper bound, **not a method** |
| `ORACLE_TIMING` best-prefix histogram | **0.2 in 20 / 20 cases** | the oracle always wants the *fewest* frames |
| `ORACLE_TIMING`, same 5-frame window | margin NORMAL +0.189 [−0.078, +0.461] vs best +0.346 [+0.102, +0.582]; paired delta **+0.158 [0.038, 0.330]** | best prefix shorter than full in **15 / 20** |
| `DELAYED` | **not implementable** | single `[SEG]` → single language prompt on the first `min(5,T)` frames → one causal propagation: there is no second commitment point, no hypothesis state to hold, and nothing a fair delayed re-initialisation could preserve |
| `ORACLE_ID` | **not implementable as a method** | the model holds exactly **one** referent hypothesis; an ID oracle would have to inject GT masks |

Headroom exists (−16.5pp) but it is entirely **oracle-only**, and its direction is
"use *less*/*cleaner* evidence", not "wait for more evidence". The architecture
offers no fair way to express `DELAYED`; per the issue's rule the pilot reports
that instead of hacking a variant that merely looks better.

### 3.5 Reproduction check

| check | result |
|---|---|
| cases compared | 20 / 20 (0 missing, 0 extra) |
| discrete fields equal | **480 / 480 (100 %)** |
| frame-level identity-error agreement | **1 909 / 1 909 (100 %)** |
| max abs difference, continuous metrics | **0.0** (target/distractor IoU, margin, IDErr) |
| re-run SHA | `4559c331cc8474320c17c7eb238f5b43a2dccb50` (4 GPUs, 4 shards) |

Every number in §3.1–§3.4 was reproduced by an independent re-inference. A
consequence worth stating: the reported identity failures are properties of the
model and its interface, not of a single stochastic sample.

## 4. The six required questions, answered

**Q1 — Is ambiguity-before-identification real and common?**
*Not answerable from the data collected so far, and the pilot says so.*
Identifiability is exactly the quantity M5.2 was built to measure, and no human
label exists yet. What the miner establishes is only that the *structural*
preconditions are common: in the stratified top-80, 78/80 cases have ≥1
same-category instance whose dilated mask overlaps the target, 58/80 have ≥2
same-category competitors, 31/80 contain target-absent annotated frames, 14/80
contain a disappearance→reappearance gap, and the median number of frames in
which target and competitor are jointly visible is 23.5. In the 250-case pool,
67 cases contain absent frames and 30 contain a gap (across all 834 mined
cases: 186 and 72).
**Assessment: structurally plausible and common; human-confirmed frequency
unknown until P2 is annotated.**

**Q2 — Does the model already make a single-object hard prediction under
ambiguous evidence?**
In the behavioural sense, yes: with only ~5 frames of evidence it already emits
a non-empty single mask in **19/20** cases (1 empty) and has already picked one
of the two competing instances (14 target / 5 distractor), with IDErr 25.5%
(frame-micro 27.4%) and only 4.2% empty frames. The commitment is then
essentially frozen (§3.3). The stronger claim — "it commits *before a human
could decide*" — requires P2 labels and is **not** established here.

**Q3 — Does the problem persist once future evidence reaches the VLM?**
Yes, and this is the most robust finding of the pilot. In the official
configuration the VLM sees **every** sampled frame inside one forward pass, so
the `[SEG]` vector is already conditioned on the whole video. Nevertheless
IDErr rises with evidence (25.5 % → 42.0 %), 0/5 wrong commitments are
corrected, and the referent vector moves by ≤4 % cosine. The failure is
therefore **not** "the model did not see the future".

**Q4 — Does the delayed/oracle analysis show enough headroom?**
There is oracle headroom of −16.5pp IDErr, but the oracle's chosen policy is
"commit from fewer frames" in 20/20 cases, and the same-window oracle gain is
about +0.158 margin. **This is headroom for a different intervention than the
one the hypothesis implies**, and the `DELAYED` intervention the hypothesis
implies cannot be implemented fairly in this architecture. So: headroom exists,
but it does **not** support "delay commitment until the evidence suffices".

**Q5 — Which formulation does the evidence support?**
**B — offline hard-referential commitment / single-hypothesis bottleneck**, with
the sharper phrasing: *the model cannot revise its referent hypothesis, and
additional evidence does not cause revision.* This is architecture-level and
code-verified: exactly one `[SEG]` per query, one language
prompt set built on the first `min(5, T)` frames, one causal propagation pass,
and no state in which a competing hypothesis could be held or compared. The
model does not merely *arrive* at one answer early — it is *structurally unable*
to revise an answer it can already see is wrong, which is why extra evidence
does not help.

**Q6 — GO or NO-GO?** → §7.

## 5. What is missing, and what it would take to close it

* **Human identifiability labels — absent.** Reason: `t*` / identifiability
  ground truth must not be produced by a model, and the pilot had no annotator
  budget. The tool and bundle are ready (P2): 80 cases, 806 checkpoints, 3
  annotators, ≈2 418 judgements (≈1.8 h per annotator; a defensible reduction
  is first/middle/last checkpoint per case = 240 judgements/annotator, keeping
  ≥60 cases).
  Entry point:
  `<artifact_root>/results/premature_commitment/annotation_bundle/annotate.html`
  (rebuild with the command in P2 §4). Labels must be `AMBIGUOUS` / `UNIQUE` /
  `INVALID` / `UNCERTAIN` (+ object id, confidence 1–5, evidence type, note),
  exported as `m5_annot_{A,B,C}.json`, then adjudicated. **With those labels the
  following become computable and are currently `N/A`:** pre-identifiability
  hard prediction rate, pre-identifiability wrong-identity rate,
  post-identifiability identity error, and the commitment/identification lag
  `Δt = t − t_identifiable`. Metric helpers for these exist and are deliberately
  unused (`metrics.pre_identifiability_hard_rate`).
* **N = 20 development cases, one model, one category.** The prefix diagnostics
  are a feasibility probe on the 4B Faithful model on `identity_swap` only; they
  are not a benchmark claim. CIs are wide (IDErr 42.0% [23.8, 59.7]).
* **No frame-level occlusion/disappearance GT** beyond the dataset's own
  annotated (every-5th) frames; "when the target is visible" is derived from GT
  mask non-emptiness.
* **Single-hypothesis claim is per-query in this code path.** Models that emit
  several `[SEG]` tokens would get several hypotheses per query by construction
  (`for seg_hidden_states in all_seg_hidden_states`), but this model emits one;
  the claim is about *this* pipeline, not about all RVOS architectures.

## 6. Threats to validity / honest caveats

1. The 20 cases come from the project's own `identity_swap` bucket, built from
   GT instance tracks. That is a legitimate, failure-independent construction,
   but it is still a constructed bucket, and it is small.
2. The paired same-window argument relies on SAM2 propagation being causal
   (verified in code: one `propagate_in_video` pass, prompt applied only on the
   first `min(5,T)` frames). It is a code-level inference supporting a measured
   effect, not an independent measurement.
3. Prefix truncation changes two things at once beyond frame `t`: the masks are
   propagated over a shorter horizon. Within the shared window this cannot
   affect the frames compared (causality), which is why the paired analysis
   exists; outside the window it must not be read as an evidence effect.
4. **Reproducibility was checked, and it passed exactly.** The audit was
   re-inferred from scratch on 4 GPUs (20 cases split into 4 shards of 5, shard
   runs at commit `4559c33`) and compared run-by-run against the committed
   reference by `projects/evoseg/premature_commitment/verify_reproduction.py`:
   **480/480 discrete fields identical** (identity decision, `has_seg`, frame
   counts), **1 909/1 909 frames with identical identity-error decisions**, and
   `max |Δ| = 0.0` on every continuous metric (target/distractor IoU, margin,
   IDErr). Evidence: `docs/premature_commitment/results/reproduction_check.json`.
5. A defect was found and repaired in the **run metadata** of the M5.0 audit:
   the committed `P0_...meta.json` recorded the repository *path* in the
   `git_sha` field, collapsed the two run modes into one global
   `vlm_all_frames=true`, and its recorded timestamp disagreed with the file
   mtime. The script is fixed (real `HEAD` SHA + per-mode flags); the committed
   metadata now records the three defects, the reconstructed run SHA
   (`730451a`, the parent of the commit that documents the run) and the exact
   reproduction above. No measured content was changed.

## 7. GO / NO-GO Decision

**NO-GO — for the "premature commitment" (prefix-causal) research direction, as
stated.**

Specifically:

1. **Formulation A is rejected.** The model is not evidence-starved: with the
   official runner it already sees every frame, `[SEG]` already conditions on
   them (cos ≤ 4 % drift as evidence grows 5×), and the system does not use that
   evidence — 0/5 wrong referents corrected, and identity gets *worse* with more
   evidence. A method whose story is "wait until the evidence is sufficient"
   has no support here. The word *premature* must not be used as a conclusion.
2. **A limited, reframed finding survives:** the pipeline is a *single-hypothesis
   bottleneck with no evidence revision* (formulation B). This is a real,
   measured, architecture-level observation, and it is the only part of the
   hypothesis the pilot is prepared to defend.
3. **Candidate formulation C is not chosen**, because the single-hypothesis half
   of the hypothesis did survive measurement (1 `[SEG]` → 1 prompt set → 1
   causal propagation; `ORACLE_ID` has nothing to choose from).
4. **Even for B, the pilot does not license method work now**, for two reasons:
   (i) the identifiability labels that would show *when* a human can decide do
   not exist, so the central "ambiguous evidence" premise is unquantified; and
   (ii) the measured oracle headroom points *away* from temporal delay, so a
   delayed/multi-hypothesis method would be motivated by the architecture
   observation rather than by the ambiguity story that was proposed.

**What would change the verdict.** Any of the following, in order of value:

* P2 annotation completed for ≥60 cases with κ reported, and
  `pre_id_hard_rate` / `pre_id_wrong_rate` / `Δt` computed. If a substantial
  fraction of checkpoints are human-`AMBIGUOUS` while the model already emits a
  confident single instance there — *and* later checkpoints become `UNIQUE` —
  then the ambiguity premise becomes real and A can be revisited with proper
  evidence.
* A second architecture that emits multiple `[SEG]` tokens (so hypotheses are
  plural by construction) showing the same identity failure — that would turn B
  from "one pipeline" into a class-level finding.
* Evidence that identity errors are caused by the VLM→prompt interface rather
  than by SAM2 propagation (e.g. prompting with GT identity in the same
  architecture, i.e. an explicit `ORACLE_ID` diagnostic) — this would localise
  the bottleneck and make it actionable.

Per the stop condition, the pilot stops here: **no belief model, no SAM3.1, no
Agent, no RL, no new verifier.**

**Next step under this verdict: none — do not start method work.** (Had the
verdict been GO, the pre-registered next step would have been: complete the P2
annotation, then run a *diagnostic-only* `ORACLE_ID` experiment to localise the
bottleneck to the VLM→prompt interface before designing anything, with any
belief/multi-hypothesis model coming only after that localization.)

## 8. Reproduction

```bash
export PYTHONSAFEPATH=1
A=/9950backfile/chenjiahui/evo_artifacts/results/premature_commitment
P=projects/evoseg/premature_commitment

# M5.0 — inference audit (GPU; 20 cases x 6 runs = 120 generations)
CUDA_VISIBLE_DEVICES=0 python $P/audit_inference_information_flow.py \
  --n-cases 20 --category identity_swap --out-dir $A

# M5.1 — candidate pool (CPU, ~8 min, deterministic)
python $P/mine_ambiguous_candidates.py --out-dir $A --pool-size 250 --top-k 80

# M5.2 — annotation bundle (CPU, ~2 min)
python $P/build_annotation_bundle.py --pool $A/candidate_pool.json \
  --out-dir $A/annotation_bundle --top-k 80 --max-checkpoints 12 --thumb-width 320

# M5.3 — diagnostics (CPU seconds, consumes the P0 JSON)
python $P/eval_prefix_identity.py --audit $A/P0_prefix_audit_identity_swap_n20.json \
  --out-dir $A/prefix_eval
python $P/eval_delayed_commit.py --audit $A/P0_prefix_audit_identity_swap_n20.json \
  --out-dir $A/headroom

# smoke test (no model/dataset/GPU; standalone, this venv has no pytest)
python $P/tests/test_metrics_smoke.py
```

Every run writes a metadata JSON (git SHA, exact command, model/dataset/manifest
paths, seed, world size, key flags). All paths are CLI-overridable; the
`/9950backfile/...` values are defaults only. No `except Exception: pass`
exists in M5 code: fallbacks print and are recorded
(e.g. `vlm_all_frames_fallback` in the audit, `matplotlib unavailable` in the
prefix eval).

## 9. Deliverable checklist (Issue #3)

| required | status |
|---|---|
| P0 inference audit; 10 questions answered from code + measurement | ✅ `P0_inference_audit.md` |
| prefix diagnostic 20/40/60/80/100 % on ~20 same-category multi-instance cases | ✅ 20 cases, `P0_...json` |
| P0 written to `docs/premature_commitment/P0_inference_audit.md`, own commit | ✅ `8feecce` |
| 150–250 real candidate cases, no toy negatives, stable keys | ✅ 250 (top-80 stratified) |
| `candidate_pool.json` / `candidate_summary.csv` / statistics / HTML | ✅ (HTML in artifact root) |
| human prefix-identifiability annotation tool, ≥3 annotators, no model labels | ✅ `annotate.html`, 806 checkpoints |
| Experiment A — prefix identity curve | ✅ |
| Experiment B — full-video vs prefix | ✅ |
| Experiment C — delayed / oracle headroom, `ORACLE_*` naming, no fake method | ✅ (DELAYED documented as not implementable) |
| frame-micro + case-macro, N/mean/median/paired delta/bootstrap CI ≥1000 fixed seed | ✅ |
| reproducible scripts, CLI-overridable paths, metadata JSON | ✅ |
| `pilot_report.md` with `## GO / NO-GO Decision` | ✅ (this file) |
| stop after pilot; no belief model / SAM3.1 / Agent / RL | ✅ respected |

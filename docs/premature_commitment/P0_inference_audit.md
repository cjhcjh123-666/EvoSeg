# P0 — Inference information-flow audit (M5.0)

**Scope**: establish, from code *and* measurement, what the current EvoSeg / Sa2VA
video inference path actually does, before any claim about "premature
commitment" is made.

**Artifacts**

| item | path |
|---|---|
| audit script | `projects/evoseg/premature_commitment/audit_inference_information_flow.py` |
| shared utils | `projects/evoseg/premature_commitment/utils.py` |
| raw diagnostics (20 identity cases) | `<artifact_root>/results/premature_commitment/P0_prefix_audit_identity_swap_n20.json` |
| run metadata | `<artifact_root>/results/premature_commitment/P0_prefix_audit_identity_swap_n20.meta.json` |

Model audited: `EvoSeg-Qwen3-VL-4B-Faithful` (image-Faithful base + external v6
temporal head), the current main-branch model.

**Metadata note (2026-09-11, M5.4).** The committed run metadata was repaired:
the original writer recorded the repository path in `git_sha` and a single
global `vlm_all_frames=true` although the file contains two run modes. The
repaired file records the three defects, the reconstructed run SHA (`730451a`,
the parent of the commit documenting this run) and the fact that every number
below was reproduced exactly by an independent 4-GPU re-inference
(`docs/premature_commitment/results/reproduction_check.json`). Measured content
is unchanged; see §6 of `pilot_report.md`.

---

**Revision note (2026-09-11, M5.5 — after review).** §0–§6 below were written on
the *gated* mask stream (`prediction_masks` = raw \* e_t) and with each prefix
averaged over its **own** horizon. Both are wrong for the questions this document
asks, and both are fixed in the schema-v2 audit:

* identity is now measured on **`raw_masks`** (pre-gate SAM2 propagation). The
  gated stream mixes referent identity with the verifier's abstention, and an
  abstained (empty) frame scores `id_err = 0`;
* every cross-prefix number now uses a strict per-case common window
  `W_i = min(sam2_prompt_frames)`, the *same* frames for both sides (revision 1
  compared 80 frames against 100 while claiming they were identical);
* `cos = 0.960` is now calibrated against a **different query on the same video**
  (0.761), and the headline number for "does more evidence help" is the
  horizon-free controlled A/B in `pilot_report.md` §3.1.

Corrected numbers (raw masks, common window, 20 cases): prefix 0.2 → 1.0 gives
`IDErr` **23.0 % → 35.0 %** (paired **+12.0 pp [0.0, +26.0]**), and the
horizon-free comparison (VLM first-5 vs VLM all-frames, same prompt frames, same
propagation) gives **25.0 % → 35.0 %** (paired **+10.0 pp [+1.0, +22.0]**). The
*direction* of every conclusion below survives; magnitudes and the
`0.960`-based phrasing are superseded. See `pilot_report.md` §3 and §7.

---

## 0. TL;DR — decision

1. The official Ref-YT-VOS runner feeds **all sampled frames to the VLM**
   (`vlm_all_frames=True`), so the default evaluation is **not** evidence-starved.
2. Giving the VLM **more** frames does **not** fix identity: on the *same* early
   frames, IDErr rises from **23.0%** (5-frame prefix) to **32.0%** (full prefix).
   **0 of 5** wrong-instance cases were corrected by the extra evidence.
3. The `[SEG]` referent vector is almost invariant to evidence volume
   (cos(prefix 0.2, full) = **0.960** mean) — the referent representation is
   **formed once and not revised**.
4. The architecture exposes **exactly one** referent hypothesis per query
   (one `[SEG]` → one prompt set → one SAM2 propagation); there is **no
   mechanism to hold or compare competing candidates**.

→ **P0 supports formulation B (offline hard referential commitment /
single-hypothesis bottleneck) and does NOT support formulation A
(prefix-causal premature commitment).** Per the issue's rule, the phrase
*"premature commitment"* is **not** used as a conclusion from here on.

---

## 1. The ten questions (code-level answers)

All line numbers are for
`projects/sa2va/hf/models_qwen3vl/modeling_sa2va_qwen.py` unless stated.

| # | Question | Answer (with reference) |
|---|---|---|
| 1 | Which video frames are visible to the VLM? | **all of them if `vlm_all_frames=True`; otherwise only the first 5** — `if (vlm_all_frames) or (frame_idx < 5): content.append({"type": "image", ...})` (`:261`). The official runner uses `vlm_all_frames=True` (`projects/evoseg/eval/infer_ryvos.py:65`, with a recorded `TypeError` fallback `:66`). |
| 2 | `vlm_all_frames=False` vs `True`? | `False` → the VLM node sees **only frames 0–4** (`:261`); `True` → all sampled frames. It does **not** change the segmentation substrate (which always receives every frame, `:290`). |
| 3 | When is `[SEG]` generated? | Inside a single greedy `self.model.generate(..., max_new_tokens=2048)` call (`:345`); the token is located afterwards via `get_seg_hidden_states(...)` (`:364`). |
| 4 | How many referent hypotheses per query? | **One per emitted `[SEG]` token.** The loop `for seg_hidden_states in all_seg_hidden_states:` (`:393`) produces one mask sequence per `[SEG]`; in practice the model emits a single `[SEG]` → **one hypothesis**. |
| 5 | Which frames produce the referent embedding? | The `[SEG]` hidden state comes from the autoregressive decode step, so it attends to **whatever frames the VLM node received** (5 or all) plus the text (`:345`–`:364`). |
| 6 | How is segmentation/tracking initialised? | All frames are encoded by SAM2 (`g_pixel_values = torch.stack([...])`, `:290`), then the language prompt is applied **on the first `num_frames = min(5, len(video))` frames** (`:294`, `:397`) via `sam2.py:332-349` (`add_language_embd(..., frame_idx, ...)`). |
| 7 | One hypothesis or several? | **One.** Prompt on the first ≤5 frames, then a single `propagate_in_video` pass over the whole video returns one mask track (`sam2.py:354-357`). |
| 8 | Can future frames influence the referent representation before masks are produced? | Yes — when `vlm_all_frames=True` every sampled frame is inside the same VLM forward, so the `[SEG]` vector is *already* conditioned on all of them. There is no sequential/streaming referent update afterwards. |
| 9 | Where is identity irreversibly collapsed? | At `:393`–`:397`: the (single) `[SEG]` vector is converted into **one language prompt set** at initialisation; everything downstream is SAM2 memory propagation, which has no mechanism to re-select an instance. |
| 10 | Which parts are offline/full-video vs sequential? | VLM = **offline over the sampled frames** (full video when the flag is set); SAM2 = **sequential causal propagation** from the first ≤5 prompted frames. |

**Structural reading**: the pipeline is `[all frames] → VLM → 1 × [SEG] →
prompt on first ≤5 frames → propagate over all frames`. Nothing in the code can
represent "object A is still possible" or "object B is still possible"; the
decision is a single hard conditioning.

---

## 2. Instrumented prefix audit

`audit_inference_information_flow.py` ran the **same 20 identity_swap cases**
(same-category multi-instance by construction) under prefix fractions
20/40/60/80/100% with `vlm_all_frames=True`, plus the default
(`vlm_all_frames=False`, full video) path. Per run it logged the decoded text,
`[SEG]` emission, the projected `[SEG]` vector, its cosine to the full-video
vector, the first non-empty frame, and per-frame target / max-distractor IoU,
identity margin, IDErr and empty flags.

### 2.1 Prefix identity curve (case-macro over 20 cases)

> ⚠️ **Superseded on two counts — see the revision note at the top.** These
> numbers come from the **gated** stream (`prediction_masks` = raw \* e_t) and
> each prefix is averaged over its **own** horizon, so a prefix with 5 frames is
> not comparable to one with 24. Corrected (raw masks, identical per-case window):
> `IDErr` 23.0 / 31.0 / 30.0 / 31.2 / 35.0 % for prefixes 0.2 → 1.0. The table is
> kept because the *direction* of every observation is unchanged and the size of
> the confound is itself informative.

| prefix | avg frames | decisions (target/distractor/empty) | IoU_t | IoU_d | margin | IDErr | cos([SEG])→full |
|---|---|---|---|---|---|---|---|
| 0.2 | 5 | 14 / 5 / 1 | 0.479 | 0.198 | **+0.280** | **25.5%** | 0.960 |
| 0.4 | 10 | 13 / 7 / 0 | 0.442 | 0.252 | +0.190 | 34.5% | 0.986 |
| 0.6 | 14 | 11 / 9 / 0 | 0.427 | 0.279 | +0.149 | 39.9% | 0.994 |
| 0.8 | 19 | 12 / 8 / 0 | 0.404 | 0.295 | +0.109 | 40.4% | 0.998 |
| 1.0 | 24 | 10 / 10 / 0 | 0.362 | 0.315 | +0.047 | **42.0%** | 1.000 |
| default (VLM = first 5, full video) | 24 | 13 / 7 / 0 | 0.457 | 0.269 | +0.188 | 34.8% | — |

Observations that matter:

* **Identity does not improve with more evidence** — it degrades over the
  horizon (IDErr 25.5% → 42.0%), and the default (first-5-frame VLM) path is
  *better* than the full-frame VLM path on this set (34.8% vs 42.0%).
* **The referent vector barely moves**: cos(`[SEG]`@0.2, `[SEG]`@full) = 0.960
  mean (min 0.787) → more evidence changes the referent representation by only
  ~4% in cosine distance.
* **Decisions are unstable across prefix length**: 5/20 = **25%** of cases flip
  identity decision between prefix 0.2 and prefix 1.0; 3/20 = 15% flip between
  the default path and prefix 1.0.

### 2.2 Is the degradation propagation drift, or the referent choice?

Scoring the **same first-5 frames** but with different evidence volume:

> ⚠️ **Superseded**: the v1 pairing did not actually align the windows (80 vs 100
> frames, because short videos have prefixes shorter than 5). The strict version
> (per-case `W_i`, raw masks) is `IDErr 23.0 % → 35.0 %`, paired
> **+12.0 pp [0.0, +26.0]**; the horizon-free version is `25.0 % → 35.0 %`, paired
> **+10.0 pp [+1.0, +22.0]** (`pilot_report.md` §3.1–§3.2).

| run | IoU_t (first 5 frames) | IDErr (first 5 frames) | decision agreement |
|---|---|---|---|
| prefix 0.2 (VLM sees only those 5 frames) | **0.493** | **23.0%** | — |
| prefix 1.0 (VLM sees all 24 frames) | **0.426** | **32.0%** | 90% vs prefix 0.2 |

Because SAM2 propagation is causal and the prompt window is always the first
≤5 frames, the *same* early frames are directly comparable — and they get
**worse** when the VLM is given more frames. Within the full run, the early
window (first 5) vs the late window (last 5) differ only mildly
(IDErr 32.0% → 35.0%), so **propagation drift is not the dominant factor**;
the referent conditioning itself is what changes (for the worse).

### 2.3 Does more evidence ever fix a wrong identity?

Same first-5-frame comparison, case level (n = 20):

| transition | count | share |
|---|---|---|
| early WRONG → late RIGHT (**evidence corrected the error**) | **0/20** | **0%** |
| early WRONG → late WRONG (error persists) | 5/20 | 25% |
| early RIGHT → late RIGHT | 13/20 | 65% |
| early RIGHT → late WRONG (more evidence hurts) | 2/20 | 10% |

**Zero corrections.** Additional video evidence never rescued a wrong referent
in this sample, and in 10% of cases it actively broke a previously correct
referent.

---

## 3. P0 decision gate

### Chosen formulation: **B — offline hard referential commitment / single-hypothesis bottleneck**

> ⚠️ **Downgraded to a *candidate* finding after review (M5.5).** B is still the
> best description of the *mechanism available in code*, but this audit does not
> establish that the single hypothesis is the **cause** of the identity errors:
> that would need an architecture that can hold multiple hypotheses, or an
> `ORACLE_ID`-style isolation diagnostic. What is measured is (i) exactly one
> hypothesis exists, (ii) the conditioning is nearly invariant to evidence volume
> (calibrated: 0.988 same-query vs 0.761 different-query), and (iii) identity does
> not improve — and in the horizon-free controlled A/B *degrades* — when the VLM
> sees more. See `pilot_report.md` §4 (Q5) and §7.

Evidence:

1. **Structural**: exactly one `[SEG]` → one prompt set → one causal SAM2
   propagation; no code path can carry two competing referent hypotheses
   (`:393`–`:397`, `sam2.py:332-357`).
2. **Behavioural**: the referent vector is nearly invariant to evidence volume
   (cos 0.960), and **0%** of wrong referents were corrected by up to 5× more
   frames — the hypothesis is formed once and not revised.
3. **Not prefix-causal**: the official path already sees all frames, and the
   full-video condition is *worse* than the 5-frame condition on the same early
   frames, which is the opposite of the prefix-causal prediction.

### Explicitly **not** chosen

* **A — prefix-causal premature commitment**: rejected. The evaluation is not
  evidence-starved, and more evidence does not help (it slightly hurts). A
  prefix-causal claim would require an online/streaming setting that this
  repository's evaluation does not implement.
* **C — hypothesis unsupported**: not chosen, because the *single-hypothesis*
  half of the statement does survive: the model cannot revise its referent, and
  errors persist. What does **not** survive is the word *"premature"*
  (i.e. a timing/insufficiency story).

**Consequence for M5**: the pilot should test whether *identity uncertainty
under ambiguity* is a real, common and consequential phenomenon — **not**
whether "delaying the commitment" helps. The delayed/oracle experiments are
re-scoped accordingly (see `pilot_report.md`), and any timing-based oracle must
be labelled `ORACLE_*`.

---

## 4. What this rules out (avoid wasted work)

* Do **not** build a "wait longer before committing" module: more evidence is
  not the missing ingredient in this pipeline.
* Do **not** describe the failure as the model "not seeing the future".
* Any future method must be justified by **evidence revision** (maintaining /
  comparing multiple referents, and being able to switch), which the current
  path structurally cannot do.

---

## 5. Reproduce

```bash
# 1 case sanity check
python projects/evoseg/premature_commitment/audit_inference_information_flow.py \
  --model <artifact_root>/models/EvoSeg-Qwen3-VL-4B-Faithful --dry-run \
  --out-dir <artifact_root>/results/premature_commitment

# full 20-case audit (identity_swap), prefixes 20/40/60/80/100%
python projects/evoseg/premature_commitment/audit_inference_information_flow.py \
  --model <artifact_root>/models/EvoSeg-Qwen3-VL-4B-Faithful \
  --manifest <artifact_root>/datasets/ref_youtube_vos/faithfulness_valid.json \
  --category identity_swap --n-cases 20 \
  --prefixes 0.2 0.4 0.6 0.8 1.0 --seed 0 \
  --out-dir <artifact_root>/results/premature_commitment
```

Run metadata (git SHA, command, model path, manifest, seed, flags) is written
next to the JSON (`P0_prefix_audit_identity_swap_n20.meta.json`). Every fallback
prints and is recorded in the JSON (`vlm_all_frames_fallback`,
`temporal_head_loaded`).

## 6. Known limitations of this audit

* 20 cases, `identity_swap` only; case-macro numbers, no bootstrap CI yet (the
  pilot adds CIs and more categories).
* The per-frame IoU uses the expression-level GT masks (verified in this
  milestone: annotation dirs are keyed by **expression id**, and expressions of
  the same object share masks) — distractors are reached through other
  expressions' obj_ids.
* No human identifiability labels yet, so "pre-identifiability" rates cannot be
  reported (that is the P2/annotation stage).

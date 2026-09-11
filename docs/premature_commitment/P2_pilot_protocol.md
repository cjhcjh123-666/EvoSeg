# M5.2 — Prefix-identifiability annotation bundle (protocol)

**Status: NO HUMAN LABELS EXIST YET.**

This document defines the annotation bundle that M5.3 needs. The bundle is
*annotation-ready only*. No label in this repository, in the artifact root, or
in any script output was produced by a model, an LLM or a heuristic. The only
state that exists today is:

| item | status |
|---|---|
| candidate pool (250 cases) | ✅ produced by mining (P1) |
| annotation subset (80 cases) | ✅ produced by stratified top-k (P1) |
| checkpoints per case (≤12) | ✅ produced by GT-event-driven placement (this stage) |
| **human identifiability labels** | ❌ **none** |

`t*` / identifiability ground truth is **not** auto-generated anywhere in M5.
Models are allowed to *select candidate cases*; they are never allowed to
produce the label.

---

## 1. What the annotator sees

For each checkpoint the tool shows

* the **query** (verbatim Ref-YT-VOS expression),
* a **filmstrip of raw frames from the video prefix** `V[1:t]` (up to 6 frames,
  evenly spaced between the first annotated frame and the checkpoint frame),
* case metadata: case index, checkpoint index, last prefix frame index, video id,
  expression id, target object id,
* *optionally* (checkbox, hidden by default) an **instance palette**: one GT-mask
  overlay thumbnail per instance id with its category. The palette exists
  **only** so that the annotator can name an object id for a `UNIQUE` verdict.
  It is off by default so that the default reading is "what can be inferred from
  the video prefix", not "what does the GT say".

The annotator **cannot** see the future frames of the video at a checkpoint.

## 2. Label schema

| label | definition |
|---|---|
| `AMBIGUOUS` | With the evidence available up to this checkpoint, **more than one** instance still satisfies the expression. |
| `UNIQUE` | The evidence available up to this checkpoint already determines **exactly one** instance. Requires an instance id. |
| `INVALID` | No legal referent exists for the expression. |
| `UNCERTAIN` | The annotator cannot decide reliably. Escalated to adjudication — never silently converted into another class. |

Per checkpoint the annotator also records:

* `selected_obj_id` — required when (and only when) the label is `UNIQUE`;
* `confidence` — integer 1–5;
* `evidence_type` ∈ {`appearance`, `action`, `temporal_order`, `relation`,
  `reappearance`, `other`};
* `note` — optional free text.

Recorded schema fields (written by the tool on export):

```json
{"case_id": "226f1e10f7:7", "video_id": "...", "exp_id": "7",
 "query": "...", "checkpoint_idx": 0, "prefix_last_frame_idx": 3,
 "state": "AMBIGUOUS|UNIQUE|INVALID|UNCERTAIN",
 "selected_obj_id": "4|null", "confidence": 1,
 "evidence_type": "appearance|action|temporal_order|relation|reappearance|other",
 "note": ""}
```

## 3. Checkpoint placement

Checkpoints are **coarse fractions** of the annotated frames
(`0.1 … 1.0` by default) plus, when the target has a disappearance gap, one
extra midpoint checkpoint. They are capped at `--max-checkpoints` (12) and
deduplicated. GT visibility/signals are used **only to place checkpoints**;
they are never shown as the answer.

We deliberately do **not** ask for a single exact `t*`. Instead the unit of
ground truth is the interval

```
identifiability interval = [ last AMBIGUOUS checkpoint , first UNIQUE checkpoint ]
```

and downstream analysis uses `Δt = t - t_identifiable`. If an annotator reports
`UNIQUE` before any `AMBIGUOUS`, or never reports `UNIQUE`, that case is
`UNCERTAIN` for the purpose of `Δt` and is excluded from Δt statistics (but
kept in the denominator and reported).

## 4. Bundle contents and how to run it

| item | path |
|---|---|
| generator | `projects/evoseg/premature_commitment/build_annotation_bundle.py` |
| tool (open in a browser) | `<artifact_root>/results/premature_commitment/annotation_bundle/annotate.html` |
| bundle JSON | `<artifact_root>/results/premature_commitment/annotation_bundle/annotation_bundle.json` |
| images | `<artifact_root>/results/premature_commitment/annotation_bundle/frames/*.jpg` |
| committed index | `docs/premature_commitment/results/annotation_bundle_index.json` |

```bash
export PYTHONSAFEPATH=1
python projects/evoseg/premature_commitment/build_annotation_bundle.py \
  --pool /9950backfile/chenjiahui/evo_artifacts/results/premature_commitment/candidate_pool.json \
  --out-dir /9950backfile/chenjiahui/evo_artifacts/results/premature_commitment/annotation_bundle \
  --top-k 80 --max-checkpoints 12 --thumb-width 320
```

Current bundle: **80 cases, 806 checkpoints, 4 817 frame JPEGs** (≈82 MB of
images; kept entirely out of git — the HTML references them relatively so the
page itself is ~0.5 MB and stays responsive).

**Workflow**

1. Open `annotate.html` in any browser (no server needed).
2. Pick `annotator: A | B | C` in the top bar (three annotators are supported;
   each has independent `localStorage` state under its own key).
3. Label every checkpoint. Keyboard: `1` = AMBIGUOUS, `2` = UNIQUE,
   `3` = INVALID, `4` = UNCERTAIN, `←`/`→` to move. A progress bar shows
   `done / total`.
4. Click **export JSON** → you get `m5_annot_<A|B|C>.json` with one row per
   checkpoint and a `state` field (`UNLABELLED` for anything skipped).
5. Send the three JSON files for adjudication.

**Human effort**: 806 checkpoints × 3 annotators = **2 418 judgements**; at
~8 s per checkpoint that is ≈1.8 h per annotator. If that is too much, the
defensible reduction is to annotate the same 80 cases with **corners only**
(first/middle/last checkpoint = 240 judgements/annotator) — but the number of
annotated cases must not drop below 60, and the choice must be stated.

## 5. Adjudication and agreement

* Compute pairwise agreement (Cohen's κ) and 3-way agreement (Fleiss' κ) over
  the 4-way label, and report raw agreement separately for the binary
  `AMBIGUOUS`-vs-`UNIQUE` frontier, which is the quantity that matters for `Δt`.
* Disagreements go to an adjudication pass; the adjudicated label is the one
  used for `Δt`. Report the adjudication rate.
* `UNIQUE` + different `obj_id` between annotators is a **conflict**, not
  agreement, and must be adjudicated explicitly.

## 6. Abort rule (pre-registered)

The mining stage is a hypothesis generator, so the annotation stage is allowed
to falsify it. Before any metric is computed, sample 30 checkpoints:

* if **> 40%** are labelled `INVALID` or `UNCERTAIN` by all annotators, the
  miner selected the wrong cases (the expression is not a resolvable referential
  ambiguity on this video). **Stop, report the failure, and do not compute
  Δt statistics** — a pool-level result of “the miner does not produce
  annotatable ambiguity” is itself a valid M5 finding.

This rule exists because an earlier manual sanity check in this project (on a
*different* construction: synthetic negative queries for the EvoSeg benchmark)
found that a majority of sampled items did not visually correspond to their
query; the same risk is checked here explicitly rather than assumed away.

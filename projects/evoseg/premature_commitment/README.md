# M5 — Premature / hard referential commitment pilot (falsification-first)

This directory implements the M5 research pilot defined in **Issue #3**:
*"Premature Referential Commitment — falsification-first pilot"*.

**Goal of M5 is diagnosis, not a new model.** No Agent, RL, controller,
belief network, new verifier or SAM3.1 integration belongs here.

> **Status: pilot complete (revision 2).**
> **NO-GO for the prefix-causal "premature commitment" hypothesis — and that is
> not a NO-GO for EvoSeg.** The surviving finding is a *candidate* one: the
> single query-level conditioning token is not turned into a better referent by
> showing the VLM more of the video (controlled A/B: IDErr 25.0 % → 35.0 %,
> paired +10.0 pp [1.0, 22.0] on identical frames and identical propagation).
> See `docs/premature_commitment/pilot_report.md` §3.1 and §7.

## Layout

```
projects/evoseg/premature_commitment/
  utils.py                              shared paths / GT masks / metrics helpers
  metrics.py                            identity + commitment metrics, bootstrap CIs
  audit_inference_information_flow.py   M5.0 prefix / information-flow audit
                                        (schema v2: raw_masks primary, gated secondary,
                                         other-query cosine calibration, sharding)
  mine_ambiguous_candidates.py          M5.1 ambiguous-referent candidate miner
  build_annotation_bundle.py            M5.2 human annotation bundle generator
  eval_prefix_identity.py               M5.3 prefix identity: strict per-case common
                                        window, controlled VLM-evidence A/B, gate check
  eval_delayed_commit.py                M5.3 delayed / ORACLE_* headroom (horizon-matched)
  verify_reproduction.py                determinism / reproduction checks
  configs/pilot.yaml                    default paths + flags for the pilot
  tests/                                smoke tests (no model download required)
```

Reports and committed results live under `docs/premature_commitment/`.
Heavy artifacts (raw masks/features, annotation HTML) live under the external
artifact root: `<artifact_root>/results/premature_commitment/`.

## Conventions (mandatory)

1. **Falsification first** — a NO-GO verdict is a valid, useful outcome.
2. **No toy negatives** (no random absent classes / category-word swaps).
3. **GT leakage must be explicit** — anything using GT timing/identity/masks is
   named `ORACLE_*` in code, filenames, tables and reports.
4. **No silent fallbacks** — every degraded path prints and is recorded in the
   run metadata; `except Exception: pass` is forbidden.
5. Every run writes a metadata JSON (git SHA, command, model/dataset/manifest
   paths, seed, world size, timestamp, key flags).
6. All machine paths are CLI-overridable; repo artifact defaults are only
   defaults.

## Quick start

```bash
python projects/evoseg/premature_commitment/audit_inference_information_flow.py --dry-run
python projects/evoseg/premature_commitment/mine_ambiguous_candidates.py --help
python projects/evoseg/premature_commitment/build_annotation_bundle.py --help
```

See `docs/premature_commitment/P0_inference_audit.md` for the inference-flow
findings, `docs/premature_commitment/P1_candidate_mining.md` for the candidate
pool, `docs/premature_commitment/P2_pilot_protocol.md` for the (still
unlabelled) annotation bundle, and `docs/premature_commitment/pilot_report.md`
for the GO / NO-GO decision.

## Reports, results and figures (committed)

| what | path |
|---|---|
| P0 inference audit | `docs/premature_commitment/P0_inference_audit.md` |
| P1 candidate mining | `docs/premature_commitment/P1_candidate_mining.md` |
| P2 annotation protocol | `docs/premature_commitment/P2_pilot_protocol.md` |
| pilot report + GO/NO-GO | `docs/premature_commitment/pilot_report.md` |
| **raw-mask P0 diagnostics (schema v2, primary)** | `docs/premature_commitment/results/P0_prefix_audit_v2_identity_swap_n20.json` |
| gated P0 diagnostics (revision 1, superseded) | `docs/premature_commitment/results/P0_prefix_audit_identity_swap_n20.json` |
| candidate pool / summary / stats | `docs/premature_commitment/results/candidate_{pool.json,summary.csv,stats.json}` |
| **prefix identity, common window + CIs (v2)** | `docs/premature_commitment/results/prefix_identity_eval_v2.json` |
| **delayed / ORACLE_* headroom, horizon-matched (v2)** | `docs/premature_commitment/results/delayed_oracle_headroom_v2.json` |
| determinism check (revision 1 == v2 gated stream) | `docs/premature_commitment/results/reproduction_check.json` |
| prefix identity curve plot (v2) | `docs/premature_commitment/figures/prefix_identity_curve_v2.png` |
| annotation bundle index | `docs/premature_commitment/results/annotation_bundle_index.json` |

Heavy per-run artifacts (raw per-frame IoUs, propagated masks, the 4 817-frame
annotation image set) are **not** committed; the generated JSON above is the
committed, sufficient summary. Every committed result file records the exact
command that produced it in its sibling `<name>.meta.json` or its `index.json`.

## Annotation entry point (human work required)

```bash
python projects/evoseg/premature_commitment/build_annotation_bundle.py \
  --pool <artifact_root>/results/premature_commitment/candidate_pool.json \
  --out-dir <artifact_root>/results/premature_commitment/annotation_bundle \
  --top-k 80 --max-checkpoints 12 --thumb-width 320
# then open <artifact_root>/results/premature_commitment/annotation_bundle/annotate.html
```

80 cases × 806 checkpoints, 3 annotators, labels
`AMBIGUOUS / UNIQUE / INVALID / UNCERTAIN` (+ object id, confidence 1–5,
evidence type, note). See `docs/premature_commitment/P2_pilot_protocol.md`.

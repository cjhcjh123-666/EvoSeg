# M5 — Premature / hard referential commitment pilot (falsification-first)

This directory implements the M5 research pilot defined in **Issue #3**:
*"Premature Referential Commitment — falsification-first pilot"*.

**Goal of M5 is diagnosis, not a new model.** No Agent, RL, controller,
belief network, new verifier or SAM3.1 integration belongs here.

## Layout

```
projects/evoseg/premature_commitment/
  utils.py                              shared paths / GT masks / metrics helpers
  metrics.py                            identity + commitment metrics, bootstrap CIs
  audit_inference_information_flow.py   M5.0 prefix / information-flow audit
  mine_ambiguous_candidates.py          M5.1 ambiguous-referent candidate miner
  build_annotation_bundle.py            M5.2 human annotation bundle generator
  eval_prefix_identity.py               M5.3 prefix identity curve + full-vs-prefix
  eval_delayed_commit.py                M5.3 delayed / ORACLE_* headroom diagnostics
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
findings and `docs/premature_commitment/pilot_report.md` for the GO / NO-GO
decision.

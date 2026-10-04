# EvoSeg — Factorized Temporal Grounding

**EvoSeg** studies Referring Video Object Segmentation with Segmentation VLMs.
The current method, **FTG**, factorizes video grounding into a persistent object
identity and frame-dependent object states before frozen SAM3.1 mask decoding.

Codebase: fork of ByteDance **Pixel-LLM (Sa2VA)** — see `README.pixel_llm.md` for the
upstream README (license preserved in `LICENSE`). Our research layer lives in
`projects/evoseg/`.

## Status (2026-10-04)

- Project specification locked: [`docs/ftg/PROJECT_SPEC.md`](docs/ftg/PROJECT_SPEC.md).
- Primary foundation: public `Sa2VA-Qwen3-VL-4B-SAM3`; scratch
  Qwen3-VL-4B + frozen official SAM3.1 is retained as a controlled ablation.
- The first additive Global/Frame/Identity/State/FTG pilot was rejected. The
  active method assigns persistent identity to native tracker memory and
  frame-dependent state to reconditioning observations.
- Earlier faithfulness and temporal-process experiments remain as archived
  diagnostics and are not part of the FTG method claim.

## Repo hygiene
- **Never commit** weights, checkpoints, data, logs, caches (see `.gitignore`).
- Raw data lives outside the repo (e.g. `/9950backfile/.../evo_artifacts`).

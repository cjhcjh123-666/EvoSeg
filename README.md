# EvoSeg — Faithful Referring Segmentation in Image & Video

**EvoSeg** builds a unified lightweight image+video referring segmenter that learns
**when to segment, when to abstain, when to verify, and when to clarify**, targeting
CVPR 2027.

Codebase: fork of ByteDance **Pixel-LLM (Sa2VA)** — see `README.pixel_llm.md` for the
upstream README (license preserved in `LICENSE`). Our research layer lives in
`projects/evoseg/`.

## Status (2026-08-13)
- Story locked: see `docs/RESEARCH_PLAN.md`.
- Base models: Qwen3-VL-4B / Qwen3-VL-8B / VILA1.5-3B (efficiency baseline).
- SFT recipe aligned with Sa2VA-3B; faithfulness GRPO + verifier + clarification in progress.

## Repo hygiene
- **Never commit** weights, checkpoints, data, logs, caches (see `.gitignore`).
- Raw data lives outside the repo (e.g. `/9950backfile/.../evo_artifacts`).

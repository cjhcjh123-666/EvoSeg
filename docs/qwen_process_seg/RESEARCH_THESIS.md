# Qwen/SAM3.1 foundation status

This directory is the audited infrastructure foundation for FTG. It establishes
exact frame-token recovery from Qwen3-VL-4B and a differentiable route into the
official frozen SAM3.1 `visual_prompt_embed` grounding interface. SAM3.1 remains
the only pixel decoder.

The earlier process-reasoner proposal is retired. No process compiler, segmental
reasoner, candidate bank, verifier, refusal mechanism, RL objective, or post-hoc
matcher is part of the active method. The current research claim and experiment
protocol are fixed in `docs/ftg/PROJECT_SPEC.md`.

The reusable evidence here is an infrastructure/capability audit, not an FTG
benchmark result.

# Implementation audit

- Base EvoSeg commit: `3699a1f124176b1ff428c1b4327e57a2c1a105d0`
- Official VIRST checkout: `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`
- Official checkout is not modified.
- Main VLM, large visual encoder, and SAM2 are frozen.
- CAPG is installed before the official frame-specific SegPrompter decoder.
- Inference capture accepts model inputs and frozen hidden states only; it has
  no GT masks, object IDs, or temporal intervals.
- Official masks/interval envelopes are loaded by a training-only module after
  the model forward and never enter inference prompt construction.

The adaptive DP has explicit PRE and POST states, forbids backward/skip
transitions, marginalizes stick-breaking process length, and exposes normalized
PRE/process/POST posteriors. The matched global-transition control shares all
language, visual, loss, and prompt components but replaces content-dependent
transition logits with learned global logits.


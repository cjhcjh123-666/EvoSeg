# Final Interface Diagnosis

## Decision: C — BOTH

Full-validation Dynamic Temporal − Static is +3.06 pp J&F (95% CI [+1.28, +5.07]) and +10.65 pp selection accuracy (95% CI [+5.92, +15.61]). The statement that temporal information is present/readable is therefore stably supported on full validation because both source-video confidence intervals are positive.

The pixel diagnosis uses the measured point/box initialization and propagation deltas in `PIXEL_EXECUTION_REPORT.md`; mask prompting is N/A for the official multiplex public API. Point prompting discards candidate-mask spatial information at initialization, while public box prompting preserves initialization substantially better; box propagation then loses most of that retained quality over the full video. Both interface stages therefore contribute material loss.

## Next Method

Design a temporal-state-to-dense-spatial-prompt interface together with memory-aware pixel execution. Use box/dense/learned mask prompts instead of a single point, and preserve the richer prompt in object memory during propagation. Do not change the VLM backbone first, because the frozen full-validation Temporal Probe shows a stable positive readout effect.

No final Method was trained in this run.

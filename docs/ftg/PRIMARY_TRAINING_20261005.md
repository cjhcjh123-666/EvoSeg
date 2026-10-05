# Main-model-first execution plan (2026-10-05)

> Historical, stopped run. Superseded by the [2026-10-06 restart](../EVOSEG_CURRENT.md).
> Do not resume this job as the new main experiment.

This supersedes the VIRST/SAM2.1 main-route decision in PROJECT_SPEC.md.
Previous negative results remain recorded in PILOT_FINDINGS.md.

## Scope and hypothesis

Task: referring video object segmentation with a segmentation VLM.
Hypothesis: a video-level identity, informed by ordered video evidence, should
condition local object state rather than letting every frame choose a referent
independently. This is a hypothesis, not a proven finding or SOTA claim.

## Implemented primary route

- Initialize Qwen3-VL-4B and its SEG projection from the trained public
  Sa2VA-Qwen3-VL-4B-SAM3 export. Do not train Qwen from scratch.
- Replace the entire pixel foundation with the official SAM3.1 multiplex
  checkpoint. No old SAM3 PVS weights are substituted into the pixel model.
- Use SAM3.1's interactive backbone, prompt encoder and mask decoder.
  A language-derived sparse prompt enters the interactive decoder, rather
  than adding a SEG vector to propagation object-validity embeddings.
- Ordered temporal evidence refines one video identity; identity-conditioned
  local spatial attention supplies dynamic state. Compose the persistent
  identity with a gated frame-local residual.
- Train final-six-layer Qwen attention LoRA, pretrained SEG projection and
  factorization. Freeze the official pixel foundation.
- Loss: mask BCE + Dice + 0.1 native object-presence BCE. GT supervises losses
  only; no GT prompt, candidate selection or visibility is used in inference.
  No Faithful, verifier, synthetic expressions or labels.

Initial main optimization uses all available expressions from public MeViS-v2
and Long-RVOS **train**. Hold out 5% of train videos, not expressions, for
development. This stage does not implement the full Public-SegMix recipe:
Ref-Youtube-VOS and image-mixture integration remains follow-up work. Training
sources inherited from the public checkpoint must also be documented before
any zero-shot or training-data-equivalence claim.

The current training path decodes sampled frames without recurrent propagation
memory. Official full-video SAM3.1 propagation integration and complete official
benchmark evaluation remain required; training losses are not benchmark J&F.
Do not advertise this stage as a finished paper method or SOTA system.

## Priority

1. Real forward/backward, loading completeness and memory checks.
2. Eight-GPU primary optimization with periodic recoverable checkpoints.
3. Official full-benchmark inference, evaluation and qualitative inspection.
4. Expand the declared public training mixture for the main model.
5. After a strong stable main result: controlled baselines and ablations.

Do not launch gate/token variants, multi-seed ablations or order shuffles now.

## Baseline reporting rule

Maintain separate provenance for reproduced and source-reported results. Use
a reproduction when its deviation is small and its protocol matches. If it is
substantially below the source report, investigate and use the source value in
the main comparison only after confirming compatibility. Retain the reproduced
value and discrepancy in notes; never select whichever value benefits FTG.

Compatibility requires the same dataset version, split, metric implementation,
aggregation, empty-target handling, frame scope and task. Disclose training
data access, external models, inference compute and resolution. MeViS val,
val_u, v1 and v2 must not be conflated. Unknown protocols remain unverified.

The local full MeViS-v2 val_u foundation reproduction is 61.95 J&F over 907
expressions, including 38 no-target expressions. The cached public model card
reports 65.3 for "MeViS (val_u)" but does not establish v2, the expression count
or empty-target handling. Until checked, 65.3 is a reference value, not a
verified like-for-like comparator.

## Entrypoint

`python -m projects.evoseg.ftg.train_primary --output RUN_DIRECTORY`

Use torchrun with eight processes after the single-process runtime check.
Runs write manifest.json, STATUS.json, training.jsonl and periodic checkpoints.
A COMPLETE status means optimization finished, not that benchmarks or the
overall research objective are complete.

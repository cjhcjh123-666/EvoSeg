# Controlled Long-RVOS + MeViS-v2 pilot

## Frozen protocol

The pilot samples eight expressions from each Long-RVOS expression type and 24
MeViS-v2 expressions, for 48 public examples total. A fixed seed creates a
video-disjoint, type-stratified 75/25 train/validation partition. Every example
uses 16 uniformly sampled full-range frames. The manifest records all source
indices, paths, expressions, dataset/type labels, and train/validation membership.

Each of the six interfaces trains for the same two epochs with identical seeds,
Qwen LoRA, optimizer, mask/query losses, and frozen SAM3.1 decoder. Validation
uses SAM3.1's predicted object-query score. J, F, J&F, present-frame J&F,
false-accept rate, and false-reject rate are retained per expression and
aggregated by dataset/type.

Each run also writes qualitative contact sheets for held-out examples. Green is
ground-truth-only, red is prediction-only (including temporal residue), and
yellow is overlap. These are mandatory sanity checks alongside aggregate J&F.

The training-only one-target assignment is also audited separately from
inference-time query ranking. If matched-query J&F is strong while
predicted-query J&F and query-selection accuracy are poor, first retain native
SAM3.1 scoring as a diagnostic control. The preregistered architectural remedy
is representation-aware query association: the persistent identity representation
scores frozen SAM object-query features, while the Frame Prompt baseline uses the
same head with a frame-varying representation. The same score is trained and used
at inference, so this is neither an oracle nor a post-hoc verifier.

## Preregistered decision

The primary contrast is FTG minus Frame Prompt on the held-out partition. The
pilot enters full training only when both conditions hold:

1. the mean J&F delta across Long-RVOS Dynamic, Long-RVOS Hybrid, and MeViS
   motion is at least **+0.01** (+1 J&F point); and
2. the Long-RVOS Static J&F delta is at least **-0.005** (no more than -0.5 point).

This small pilot is a direction gate, not a benchmark result and not paper
evidence. If it fails, inspect identity drift, mask/query selection, and
qualitative predictions before altering the method. Do not silently change the
threshold or story after observing results.

Passing the numerical gate is necessary but not sufficient to claim the
hypothesis is supported. The summary separately flags whether all three
motion-sensitive groups improve in the same direction, whether their mean gain
exceeds the Static gain, and whether any validation group has fewer than five
examples. Mixed directions or tiny groups require a larger replication before
full training even when the preregistered arithmetic gate says `GO`.

Temporal order (Original/Shuffle/Reverse) and T=8/16/32 are diagnostics after the
primary controlled run; they are not substitutes for the primary contrast.

Launch all six primary controls on six GPUs with:

```bash
python -m projects.evoseg.ftg.launch_controlled_pilot \
  --output /path/to/new/run --gpus 0 1 2 3 4 5
```

The launcher writes one shared manifest before starting any worker and records
the physical GPU, PID, command, and log path in `launch_state.json`.

# Minimal Temporal Matcher Prototype

## Outcome

The gate-authorized minimal prototype completed, and its core claim **failed**: on the held-out Dynamic split, the parameter-matched temporal matcher (D) scored J&F 0.4291 versus 0.4468 for the static scorer (C). The three-seed object-weighted difference was **D−C = −1.77pp**, with 95% source-video cluster bootstrap CI **[−4.51,+0.64]pp**.

Therefore this implementation does not demonstrate useful temporal sequence modeling. No larger training was started, and the one-epoch smoke result was not used for model or seed selection.

## Frozen-component protocol

- Source objects/expressions: unchanged 274-object Long-RVOS paired manifest and official expressions.
- Split: source-video-disjoint 60 train / 15 validation / 15 test videos, deterministic seed=42. The test set has 40 objects and 188 expressions.
- Query: cached Sa2VA N=16 `z_seg` (256 dimensions); Sa2VA was not rerun or updated.
- Candidate bank: cached official SAM3.1 deterministic-concept tracks; SAM3.1 was not rerun or updated.
- Track appearance: official InstructSeg checkpoint's frozen SigLIP So400m patch features, pooled within each candidate mask at 8 ordered, uniformly spaced evaluation positions. All 448 vision tensors loaded exactly, with zero missing/unexpected keys.
- GT separation: feature extraction read no GT. GT entered only after extraction to identify the best candidate target for official training supervision.
- Candidate hit for training: oracle J&F≥0.3. Candidate misses were not assigned an incorrect positive. There were 738 training and 162 validation expressions after this filter.
- C and D both have 462,337 trainable parameters and the same 2-layer Transformer/scoring stack. C averages the eight visual features before the stack; D retains their order and adds fixed sinusoidal positions. Geometry is not a primary feature.
- Seeds: 11, 23, 42. AdamW, maximum 80 epochs, patience 10. Each seed was completed and averaged; no seed was selected by its test result.

## Main test results

Scores are object-weighted within description type. Candidate-generation failures or zero candidates score zero; other candidate misses remain marked and receive the J&F of the actually selected candidate.

| Condition | Static J&F | Dynamic J&F | Hybrid J&F |
|---|---:|---:|---:|
| A. SAM3.1 raw expression | 0.2992 | 0.1920 | 0.2170 |
| B. SAM3.1 deterministic concept only | 0.4082 | 0.3806 | 0.4180 |
| C. concept + static track scorer | **0.5001** | **0.4468** | **0.4611** |
| D. concept + temporal track matcher | 0.4743 | 0.4291 | 0.4506 |
| concept candidate oracle (upper bound) | 0.6293 | 0.5933 | 0.5709 |

A/B select the highest available official confidence; if the official tracker supplies no confidence for every candidate, they use its deterministic first output. This rule never reads GT. The oracle is reported only as candidate-bank headroom, not as a selectable method.

The held-out candidate records were complete (188/188 successful). At the J&F≥0.3 training-hit definition, expression counts were 62/67 Static, 56/64 Dynamic, and 47/57 Hybrid. These misses were retained in test evaluation.

## Core ablation stability

| Seed | Dynamic D−C J&F | 95% source-video bootstrap CI | Static/temporal early-stop epochs |
|---:|---:|---:|---:|
| 11 | −3.48pp | [−8.97,+1.57]pp | 13 / 11 |
| 23 | −3.98pp | [−6.24,−1.75]pp | 14 / 12 |
| 42 | +2.13pp | [−1.70,+6.68]pp | 16 / 12 |
| three-seed mean | **−1.77pp** | **[−4.51,+0.64]pp** | — |

The sign changes across seeds, and the prespecified three-seed aggregate is negative with an interval crossing zero. Consequently `D > C`, especially on Dynamic expressions, is not established.

## Interpretation and stop decision

The cross-model diagnosis still selects Route B as the research direction: VIRST's stronger temporal organization substantially reduced the full 274-object Dynamic gap. This minimal SAM3.1 track-matcher instantiation, however, is a **NO-GO** as the claimed method. Possible unresolved causes include the temporal information content of region-pooled frozen SigLIP features, the use of one Sa2VA `z_seg` query state, and candidate identity/visibility across the eight pooled frames. The current test does not distinguish those causes.

Per the decision gate, no larger matcher, VLM fine-tuning, SAM3.1 fine-tuning, geometry-heavy alternative, or additional seed sweep was launched. Any next experiment requires review and should first audit track-aligned feature content rather than scale this negative configuration.

## Artifacts

- Full artifact root: `/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260927_temporal_matcher_prototype`
- `results/matcher_prototype_summary.csv`
- `results/matcher_prototype_per_seed_summary.csv`
- `results/matcher_prototype_test_predictions.csv`
- `results/matcher_prototype_audit.json`
- `results/matcher_training_history_seed{11,23,42}.json`
- `figures/matcher_prototype_summary.png`
- `figures/matcher_prototype_fixed_cases.png` and its prediction-independent selection audit JSON

The `.pt` features, candidate RLE masks, videos, and checkpoints remain only in the artifact store and are not committed to git.

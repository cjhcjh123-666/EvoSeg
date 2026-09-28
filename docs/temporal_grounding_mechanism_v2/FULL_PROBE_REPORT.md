# Full Temporal Probe Validation

The preregistered, parameter-identical probes were evaluated on all 274 paired validation objects without validation training, seed selection, architecture changes, or hyperparameter changes. 1188/1189 expressions produced evaluable frozen inputs; 1 deterministic state-extraction failure(s) are listed below and were not imputed. The 4 official candidate-generation failure(s) were retained as genuine no-candidate outcomes (zero candidate-direct J&F), rather than silently removed. The original implementation did not persist pilot checkpoints; the three probes were therefore deterministically reconstructed once from the unchanged official-train tensors/split/config, checked against the pilot outputs, frozen, and loaded in `load_only` mode for this full evaluation.

| description_type | method | expressions | candidate_hit_expressions | candidate_miss_expressions | selection_accuracy_on_candidate_hits | candidate_direct_J_and_F |
|---|---|---|---|---|---|---|
| static | static | 425 | 383 | 42 | 48.83 | 45.37 |
| static | temporal | 425 | 383 | 42 | 58.20 | 48.36 |
| dynamic | static | 417 | 385 | 32 | 42.94 | 43.79 |
| dynamic | temporal | 417 | 385 | 32 | 53.59 | 46.85 |
| hybrid | static | 346 | 296 | 50 | 49.57 | 42.32 |
| hybrid | temporal | 346 | 296 | 50 | 59.35 | 45.25 |

## Dynamic paired effect

- Selection accuracy Temporal − Static: **+10.65 pp**, source-video bootstrap 95% CI **[+5.92, +15.61] pp** (255 hit objects, 89 videos; 2,000 resamples).
- Candidate-direct J&F Temporal − Static: **+3.06 pp**, source-video bootstrap 95% CI **[+1.28, +5.07] pp** (273 objects, 90 videos; 2,000 resamples).

## Preregistered seeds

| model_seed | method | selection_accuracy_on_candidate_hits | candidate_direct_J_and_F |
|---|---|---|---|
| 11 | static | 42.75 | 43.70 |
| 11 | temporal | 52.94 | 46.84 |
| 23 | static | 42.55 | 43.52 |
| 23 | temporal | 52.16 | 46.05 |
| 42 | static | 42.16 | 43.51 |
| 42 | temporal | 50.98 | 45.80 |

## State extraction failures

- `long_rvos/0102096b4c/1/2`

## Candidate-generation failures (counted as misses)

- `long_rvos/939b6aed6d/2/5`
- `long_rvos/939b6aed6d/3/8`
- `long_rvos/939b6aed6d/3/9`
- `long_rvos/f3fdaa2e8b/2/10`

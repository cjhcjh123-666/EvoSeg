# Failure analysis

Counts below follow fixed rules over every ensemble result; no cases were dropped based on outcome.

| Dataset | paired | candidate miss | OPG fixes mean-pool | OPG damages mean-pool | both correct | both wrong |
|---|---:|---:|---:|---:|---:|---:|
| Long-RVOS | 1188 | 124 | 0 | 0 | 658 | 530 |
| GroundMoRe Sequential | 480 | 425 | 0 | 0 | 40 | 440 |

For verified GroundMoRe target tracks, Full OPG scores original above reverse in 39/55 cases and above block-swap in 43/55 cases. Candidate misses remain an executor ceiling and are always scored as failures. `figures/qualitative_order_cases/` contains deterministic success, bag-of-frames failure, static-cue, and candidate-miss examples rather than success-only curation.

Official metadata/archive availability failures are retained separately: trainval 2, test 0. They are not converted into positive candidates or silently counted as successful training samples.

# CPG-VIRST Overfit Capability Gate

Status: **FAIL** (576/576 updates)

## Measurements

- Segmentation loss, early → late: `1.1026322186226025` → `1.5880067136604339` (delta `0.48537449503783137`)
- Per-question first → last segmentation loss: mean delta `-0.4295914240065031`, decrease rate `0.59375` over `64` repeated questions
- Original > fixed permutation rate: `1.0` over `61` unique verified-order samples
- Target > best distractor rate: `0.9666666666666667` over `30` unique samples with official distractor masks
- Interval localization: `N/A`; GroundMoRe's official interval is the whole queried-process/mask-validity span, not an unambiguous target-clause label
- Expected process length: `3.518842164427042`
- Mean state duration: `1.3327734097838402` frames
- Posterior entropy: `1.0201124716550112`
- Posterior expected-state monotonic rate: `1.0`
- Posterior non-single-state-collapse rate: `1.0`

## Preregistered checks

- PASS: `segmentation_loss_decreased`
- FAIL: `interval_localization_improved`
- PASS: `original_gt_reverse_rate_at_least_90pct`
- PASS: `target_gt_distractor_rate_at_least_85pct`
- PASS: `posterior_non_degenerate`

Pilot launch authorized: **False**.

The unavailable interval label is not replaced by a pseudo-label. Consequently
the full preregistered capability gate cannot pass under the current official
metadata, even if the other training checks succeed.

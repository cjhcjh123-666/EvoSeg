# Final CPG-VIRST SFT decision

Decision: **NO-GO under the requested SFT protocol**.

The capability run completed all 576 fixed updates and passed four measurable
checks: paired segmentation loss decreased; original exceeded its fixed
permutation on 100% of resolvable examples; target exceeded distractors on
96.67% of usable examples; and the posterior was monotone/non-degenerate.

The mandatory interval-localization check cannot be computed from official
GroundMoRe annotations without treating a whole-process mask-validity interval
as a target-clause label. No pseudo-label was created. The Overfit Gate is
therefore FAIL and explicitly sets `pilot_authorized=false`.

Accordingly:

- no GroundMoRe or Long-RVOS CPG pilot was launched;
- no full evaluation, ablation expansion, or seeds 23/42 were launched;
- no RL design or training was started;
- baseline values remain references only, with no invented CPG delta.

This is not an `ARCHITECTURE NO-GO` based on failure to fit the measurable
training objectives: order, object discrimination, segmentation loss, and
posterior continuity did fit. It is a **supervision/protocol NO-GO** because one
preregistered capability target is unavailable and variable-length behavior
also remains weak (argmax N=5 for all 64 examples). The near-identical posterior
shape across examples is an additional mechanism warning: continuity is real,
but content-sensitive duration alignment is not established.

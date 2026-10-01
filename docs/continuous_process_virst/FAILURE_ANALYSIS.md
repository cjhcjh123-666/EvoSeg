# Failure analysis

## Decisive blocker: supervision semantics

The requested `L_loc` assumes that GroundMoRe `action_start/action_end` labels
the language answer clause. The exhaustive deterministic audit contradicts
that assumption: 442 interval groups reuse one interval for questions with
different answer clauses, including 122 groups mixing before/after orientation.
The interval is suitable for official temporal-mask evaluation but not for
assigning one latent process state to a target clause.

Using it as a clause-local label would silently introduce a false supervision
target. The implementation therefore leaves `L_loc` disabled and records that
fact in every training row and checkpoint. This single missing mandatory check
is sufficient to fail the preregistered gate.

## What did work on the training capability set

- Per-expression segmentation loss decreased by 0.4296 on average.
- Original sequence beat its fixed permutation on 61/61 resolvable examples.
- Target object beat the best official distractor on 29/30 usable examples.
- Segmental posteriors were monotone and non-single-state-collapsed on 64/64.

## Remaining mechanism warning

Although the expected process length was 3.5188, `argmax P(N|q)` was 5 for every
expression and the termination entropy was nearly maximal. The variable-length
head therefore has not demonstrated query-adaptive length. Moreover, posterior
cell values vary by only 0.0171 across expressions while compatibility cells
vary by 0.2428: the plotted posteriors are nearly the same smooth transition
template. The structure enforces a legal forward process, but it has not shown
content-sensitive state durations. Overfit success on order and object losses
does not establish validation generalization.

The outcome is a protocol-level SFT NO-GO, not proof that continuous segmental
grounding is impossible. A future attempt needs an official dataset whose
interval annotation is explicitly tied to the answer-bearing language clause,
or a reformulated non-clause-local objective preregistered before evaluation.

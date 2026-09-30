# GroundMoRe candidate-generation improvement audit

This is an inference-preprocessing audit, not an OPG-v2 contribution. The deterministic parser reads official question text only. It maps person interrogatives to broad `person`, uses an explicit parsed noun phrase when available, and leaves unresolved questions without a prompt. GT object IDs, answers, and masks are never read during prompt construction.

| condition | evaluated / 480 | nonempty | Recall@0.3 | Recall@0.5 | Recall@0.7 | oracle J&F (all 480) |
|---|---:|---:|---:|---:|---:|---:|
| Raw full expression | 480 / 480 | 88 | 11.46% | 5.83% | 2.71% | 6.72 |
| Deterministic concept | 472 / 480 | 462 | 65.42% | 48.33% | 25.00% | 43.18 |

Unresolved or failed deterministic prompts: 8. Candidate-generation changes are reported separately and are not attributed to ordered reasoning.

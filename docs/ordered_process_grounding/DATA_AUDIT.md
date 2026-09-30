# Data audit

- Long-RVOS official-train cache: 312 expressions, 280 candidate hits; deterministic connector filter yields 120 fitting expressions over 89 videos.
- Long-RVOS paired validation: 1188 expressions; 226 deterministic order-word hits.
- GroundMoRe source: official author repository commit `d5074ab920a86a7ea92ecc69902109f6887f3e10` and official v2 metadata. Trainval has 1173 official Sequential expressions: 1171 available and 2 explicitly retained metadata/archive failures. Held-out official test has 480 expressions: 480 available and 0 failures.
- GroundMoRe GT audit: trainval visible/all-zero = 1167/4; test visible/all-zero = 480/0. Same-stem v2 compatibility is used for 2 trainval and 0 test expressions. All-zero sampled GT never creates an oracle positive.
- Classification uses only GroundMoRe's official `q_type=Sequential` or the preregistered Long-RVOS connector regex. No LLM/manual event labels, new queries, or pseudo labels were used.
- GT is used only for candidate-to-target assignment and evaluation, never candidate generation or inference scoring.

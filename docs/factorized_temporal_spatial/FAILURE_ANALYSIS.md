# Failure analysis

- Missing temporal-state expressions: 1: `long_rvos/0102096b4c/1/2`.
- Candidate-generation failures: 4: `long_rvos/939b6aed6d/2/5`, `long_rvos/939b6aed6d/3/8`, `long_rvos/939b6aed6d/3/9`, `long_rvos/f3fdaa2e8b/2/10`.
- Candidate misses are retained with zero end-to-end output; no false positive label is manufactured.
- The missing-state expression is retained as a paired zero output for identity selectors, so it cannot improve the reported delta by deletion.
- Official native confidence is unavailable for 5 raw-expression and 4 concept records; those baseline cells are reported N/A and excluded from that baseline's mean.

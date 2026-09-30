# Full validation results

| Type | Method | Evaluated | Sel. acc. on hits | J | F | J&F |
|---|---|---:|---:|---:|---:|---:|
| static | SAM3.1 raw expression | 421/425 | 61.27 | 32.55 | 32.88 | 32.71 |
| static | SAM3.1 concept native | 424/425 | 44.73 | 45.76 | 46.81 | 46.29 |
| static | Static identity | 425/425 | 48.83 | 44.93 | 45.81 | 45.37 |
| static | Temporal identity (FTSG) | 425/425 | 58.20 | 47.85 | 48.87 | 48.36 |
| static | ORACLE candidate identity | 425/425 | 100.00 | 66.36 | 66.95 | 66.66 |
| dynamic | SAM3.1 raw expression | 418/418 | 56.62 | 22.22 | 22.90 | 22.56 |
| dynamic | SAM3.1 concept native | 417/418 | 36.60 | 42.12 | 42.98 | 42.55 |
| dynamic | Static identity | 418/418 | 42.77 | 43.26 | 43.99 | 43.63 |
| dynamic | Temporal identity (FTSG) | 418/418 | 53.39 | 46.14 | 47.22 | 46.68 |
| dynamic | ORACLE candidate identity | 418/418 | 100.00 | 67.55 | 68.07 | 67.81 |
| hybrid | SAM3.1 raw expression | 345/346 | 56.05 | 24.55 | 25.04 | 24.79 |
| hybrid | SAM3.1 concept native | 344/346 | 47.02 | 41.95 | 42.72 | 42.34 |
| hybrid | Static identity | 346/346 | 49.57 | 42.03 | 42.61 | 42.32 |
| hybrid | Temporal identity (FTSG) | 346/346 | 59.35 | 44.87 | 45.64 | 45.25 |
| hybrid | ORACLE candidate identity | 346/346 | 100.00 | 61.99 | 62.66 | 62.32 |

## Dynamic primary comparison

- Selection accuracy: +10.61 pp, 95% CI [+5.92, +15.50].
- Candidate-direct J&F: +3.05 pp, 95% CI [+1.28, +5.04].
- Coverage: 274 paired objects, 1189 official expressions; 1 state failure retained as a zero-output paired failure.

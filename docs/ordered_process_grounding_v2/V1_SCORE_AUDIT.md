# OPG-v1 score audit

The audit reuses the saved three-seed v1 ensemble and frozen evaluation features; no model is retrained.

| Subset | expressions | base std | order std | effective residual std | residual/base ratio | top-1 flip | margin before | margin after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 402 | 2.91729 | 0.09722 | 0.00100 | 0.00050 | 0.00% | 0.61541 | 0.61559 |
| dynamic | 227 | 3.03598 | 0.10609 | 0.00109 | 0.00053 | 0.44% | 0.72685 | 0.72690 |
| hybrid | 273 | 2.82693 | 0.09500 | 0.00098 | 0.00045 | 0.00% | 0.73308 | 0.73321 |
| explicit_order | 214 | 3.32035 | 0.10413 | 0.00107 | 0.00049 | 0.00% | 0.75032 | 0.75042 |
| groundmore | 88 | 0.25371 | 0.03856 | 0.00040 | 0.00140 | 0.00% | -0.01230 | -0.01194 |

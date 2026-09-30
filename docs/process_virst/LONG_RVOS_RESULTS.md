# Long-RVOS result status

The frozen official VIRST baseline prerequisite passed on all 1,189 official
paired-validation expressions:

| Type | Objects | J | F | J&F |
|---|---:|---:|---:|---:|
| Static | 274 | 59.24 | 59.50 | 59.37 |
| Dynamic | 274 | 58.75 | 59.25 | 59.00 |
| Hybrid | 263 | 58.76 | 59.26 | 59.01 |

Dynamic minus Static is -0.37 pp, 95% CI [-2.68,+1.83] pp, reproducing the
prior audited run exactly.

ProcessVIRST was not evaluated on the planned Long-RVOS 64-video pilot because
the earlier GroundMoRe decision gate had already failed: normal segmentation
gain was negligible and three-seed order preference was centered at zero with
inconsistent signs. The preregistered rule says to stop in this condition rather
than continue to larger validation or tune the SFT. Accordingly, ProcessVIRST
Static/Dynamic/Hybrid values are **N/A (early stop)**, not zero and not missing
silently.

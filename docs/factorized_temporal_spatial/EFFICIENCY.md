# Efficiency

| Component | Params/head | Ensemble | Total selector params | Latency/query (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| sam31_concept_candidate_generation | frozen | 1 | frozen | 82.3445 | 34.34 |
| static_identity_selector_ensemble | 288257 | 3 | 864771 | 0.0110 | 0.03 |
| temporal_identity_selector_ensemble | 288257 | 3 | 864771 | 0.0042 | 0.03 |
| temporal_order_shuffle_ensemble | 288257 | 3 | 864771 | 0.0043 | 0.03 |
| temporal_mean_pool_ensemble | 287777 | 3 | 863331 | 0.0039 | 0.02 |
| ftsg_total_candidate_plus_selector | 288257 | 3 | 864771 | 82.3487 | 34.34 |

Candidate generation is reported separately and is not hidden inside selector-only overhead.

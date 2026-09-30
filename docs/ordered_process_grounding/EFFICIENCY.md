# Efficiency

- OPG trainable parameters per seed: 1,289,218; three-seed ensemble loaded parameters: 3,867,654.
- GroundMoRe official SAM3.1 raw-expression candidate generation: 1649/1651 successful; synchronized mean/median/p95 latency 21.709/21.336/33.380 seconds per expression; observed peak 23.83 GiB.
- Candidate generation and selector timing are reported separately. Concurrently contended GPU wall time is not presented as a clean deployment benchmark.

| Dataset | component | synchronized ms/query | peak allocated GiB |
|---|---|---:|---:|
| long_rvos | frozen_input_preparation | 53.871 | 0.696 |
| long_rvos | mean_pool | 0.075 | 0.695 |
| long_rvos | opg_no_monotonic | 2.138 | 0.697 |
| long_rvos | opg_no_order_loss | 2.237 | 0.697 |
| long_rvos | opg_full | 2.248 | 0.697 |
| groundmore | frozen_input_preparation | 1.659 | 0.714 |
| groundmore | mean_pool | 0.017 | 0.713 |
| groundmore | opg_no_monotonic | 0.427 | 0.714 |
| groundmore | opg_no_order_loss | 0.440 | 0.714 |
| groundmore | opg_full | 0.450 | 0.714 |

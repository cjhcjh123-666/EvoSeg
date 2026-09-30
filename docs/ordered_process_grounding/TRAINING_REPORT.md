# Training report

Seeds are fixed at 11/23/42; no final-validation seed selection or hyperparameter sweep was performed. Stage 1 is exactly one identity-only epoch, followed by fixed-budget ordered SFT with train-internal source-video early stopping.

- opg_no_monotonic seed 11: 34.6s, alpha=0.0116, params=1,289,218
- opg_no_monotonic seed 23: 24.8s, alpha=0.0106, params=1,289,218
- opg_no_monotonic seed 42: 22.0s, alpha=0.0101, params=1,289,218
- opg_no_order_loss seed 11: 33.1s, alpha=0.0115, params=1,289,218
- opg_no_order_loss seed 23: 35.5s, alpha=0.0115, params=1,289,218
- opg_no_order_loss seed 42: 39.9s, alpha=0.0117, params=1,289,218
- opg_full seed 11: 24.4s, alpha=0.0101, params=1,289,218
- opg_full seed 23: 30.7s, alpha=0.0105, params=1,289,218
- opg_full seed 42: 27.8s, alpha=0.0100, params=1,289,218

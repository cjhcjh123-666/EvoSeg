# Efficiency

Capability-run measurements on one A800 80 GB:

- Trainable parameters: 3,291,917
- Updates: 576
- End-to-end wall time including model initialization/checkpoint save: 1,805.85 s
- Peak CUDA allocated memory: 19.587 GiB
- Mean end-to-end time per update including initialization amortized: 3.14 s
- CPG residual beta: 0.5013 at completion

These are training capability measurements, not isolated inference benchmarks.
Pilot inference was not authorized, so selector-free CPG inference latency and
full evaluation memory are N/A. No efficiency claim is made.

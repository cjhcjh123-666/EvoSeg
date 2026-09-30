# Failure analysis

- Retained historical failed attempts: 745; these were CUDA OOM attempts and were not silently dropped.
- The failed pairs were retried with one SAM3.1 worker per GPU, followed by exclusive targeted retries for the final four pairs.
- Final successful identity/condition pairs: 1100/1100; missing: 0.
- Successful dense stage records: 3300.
- Qualitative selection is deterministic from Dynamic Temporal-P8 minus Static-P8 and includes success, near-zero, and damage cases.

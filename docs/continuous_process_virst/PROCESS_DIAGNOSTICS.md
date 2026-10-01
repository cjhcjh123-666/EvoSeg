# Process diagnostics

Diagnostics below are from the last occurrence of each of the fixed 64
capability expressions.

- Expected process length: 3.5188 states
- Argmax termination length: N=5 for 64/64 expressions
- Mean termination entropy: 1.7892 versus `log(6)=1.7918`
- Mean maximum termination probability: 0.1904
- Mean state duration: 1.3328 sampled frames
- Mean duration by slot: [1.6938, 1.8135, 1.7496, 1.4823, 1.0118, 0.2456]
- Mean frame-posterior entropy: 1.0201
- Monotone expected-state trajectories: 64/64
- No single state occupied at least 90% of frames: 64/64
- Mean cross-expression posterior cell standard deviation: 0.0171
- Mean cross-expression compatibility-cell standard deviation: 0.2428
- Final beta: 0.5013

The posterior is continuous and respects stay/advance-one transitions; it did
not collapse all frames into one state. However, the termination prior remains
almost maximally entropic and its tiny modal preference is always N=5. Thus the
run did not learn convincing query-dependent process length despite a sensible
expected N after marginalization. This is retained as a negative diagnostic.

Visual inspection also shows that the posterior path has almost the same smooth
left-to-right template for different videos and queries. Its mean cellwise
cross-expression standard deviation is only 0.0171 even though the underlying
compatibility matrices vary much more (0.2428). Thus it passes the literal
continuity/non-single-state checks, but much of that behavior is imposed by the
transition structure rather than learned content-sensitive duration alignment.

Per-expression values are in `process_diagnostics.csv`; the deterministic
six-case posterior plot is `process_posterior_examples.png`.

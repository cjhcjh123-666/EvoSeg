# ProcessVIRST SFT method (implementation record)

ProcessVIRST tests whether explicit ordered-process reasoning helps VIRST before
frame-specific segmentation prompts are formed. It keeps the official VIRST
backbone and SAM2 pixel executor and introduces only a small pre-prompt module.

For frozen query hidden states `Q`, four learned ordinal slots cross-attend `Q` to
form process states `E`. Each slot attends the official SAM2 spatial tokens at
each frame, producing event-frame states `r[m,t]` and compatibility matrix
`A[m,t]`. An exact log-sum-exp dynamic program marginalizes strictly increasing
alignments, including validity-gated slot skips. Its posterior `P[m,t]` produces
the per-frame context `c[t] = sum_m P[m,t] r[m,t]`.

The context is fused as `LayerNorm(h[t] + beta Adapter(c[t]))` immediately before
the official SegPrompter decoder. `beta` is positive by softplus construction and
is initialized to 1.0. The resulting frame-specific prompts enter the unchanged
official VIRST/SAM2 executor.

The first SFT configuration freezes the VLM, SAM2, and large visual encoder. It
trains the Process Compiler, event-frame projection, monotonic alignment path,
adapter, beta, and—only if required by the preregistered run—SegPrompter parameters
or a lightweight adapter. Official VIRST segmentation losses are retained. On
verified sequential samples, a fixed identity-hashed reverse or half-block-swap
negative adds `relu(0.2 - A_original + A_permuted)` with weight 1.0.

No RL is started in this branch unless a completed SFT run passes the stated gate;
even then, only an RL design document is permitted.

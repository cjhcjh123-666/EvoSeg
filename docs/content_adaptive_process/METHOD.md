# CAPG-VIRST method

CAPG-VIRST replaces CPG-v1's whole-video fixed left-to-right template with a
content-adaptive latent chain:

`PRE → p1 → … → pN → POST`.

The process states are weighted sums of frozen VIRST language token states,
plus ordinal embeddings and a small residual MLP. Prefix-consistent
stick-breaking predicts `P(N|q)`. Start, stay, advance, and end probabilities
are functions of the language state, spatially attended VIRST frame features,
and consecutive-frame feature deltas. A log-space forward-backward algorithm
marginalizes every legal length and path.

The process occupancy posterior is weakly supervised by GroundMoRe's official
whole-process/mask-validity interval. Posterior-pooled visual segments receive
an InfoNCE language alignment objective. Training-only official instance masks
retain object-relative process discrimination. The posterior-weighted semantic
state, occupancy, and transition confidence enter immediately before the
official VIRST SegPrompter; the official SAM2 executor is unchanged.


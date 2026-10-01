# Data audit

The capability set is exactly the CPG-v1 fixed training-only pool:

- 32 GroundMoRe official Sequential expressions from 31 videos;
- 32 Long-RVOS explicit-order expressions from 24 videos;
- deterministic `sha256(CPG-OVERFIT-42::video/expression)` selection;
- exact 64-expression interleave per epoch;
- no validation/test examples and no generated labels or queries.

GroundMoRe missing mask PNGs outside the official action/mask-validity window
define the whole-process envelope for the already sampled frames. This target
supervises only total process occupancy `rho_t`; it is never interpreted as an
event/slot boundary.


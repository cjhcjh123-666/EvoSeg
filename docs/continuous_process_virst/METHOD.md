# CPG-VIRST method contract

CPG-VIRST inserts continuous process reasoning inside VIRST, immediately before
the official frame-specific `SegPrompter`. It does not use a frozen downstream
selector, candidate-track post-processing, SAM3.1, point/box prompts, or RL.

The compiler cross-attends six learned ordinal queries to the frozen language
token sequence and predicts a prefix-consistent termination distribution
`P(N|q)`. Each process state spatially cross-attends the official VIRST/SAM2
feature map at every input frame. A log-space segmental forward–backward layer
marginalizes paths that start at state 1 and may only stay or advance by one.
Thus an active state can span several frames, backward transitions and skipped
states are impossible, and every state up to terminal state `N` is visited.

The posterior produces a per-frame state
`c_t = sum_m gamma[t,m] W_c p_m`. A two-layer adapter and positive residual scale
(`beta=0.5` initially) condition the official segmentation state before the
unchanged official SegPrompter decoder and SAM2 executor.

Only the compiler, projections, segmental transition parameters, adapter, beta,
and fusion normalization are trainable in the capability run. VIRST's VLM,
large visual encoder, official SAM2 executor, and official checkout are frozen.

The training objectives are gated by available official supervision:

- official VIRST segmentation loss on all samples;
- object-relative process CE only when official training instance masks provide
  a target plus distractors;
- fixed reverse/half-block order contrast only for verified ordered samples;
- weak termination-entropy regularization with weight `0.01`.

The GroundMoRe audit shows that `action_start/action_end` annotates the whole
queried-process/mask-validity window rather than one language clause. It is not
used as clause-level `L_loc`; doing so would manufacture a label. This makes the
requested interval part of the overfit gate explicitly unavailable unless an
official clause-level annotation is found.

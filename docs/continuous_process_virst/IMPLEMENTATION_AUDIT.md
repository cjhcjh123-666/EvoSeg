# Implementation audit

## Immutable bases

- EvoSeg base: `ad1f698f1455f3ab78bd560c03f17393fbb7cc3f`
- Official VIRST checkout: `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`
- Official checkout status at implementation time: clean
- VIRST checkpoint: `external/VIRST/checkpoints/virst_checkpoint.pt`
- Pixel executor: official VIRST SAM2.1 Hiera Large checkpoint and code

No file in the official VIRST checkout was edited. The overlay is installed by
wrapping the initialized SegPrompter from EvoSeg code.

## Verified insertion point

Official VIRST extracts final `[SEG]` hidden states and calls
`model.seg_prompter(seg_token=..., image_token=images_sam_feat, ...)`. The wrapper
uses those same official SAM/video spatial features for process–frame emissions,
adds the posterior-conditioned residual to the projected per-frame segmentation
state, and then executes the official SegPrompter decoder. This is before frame
score/anchor selection, SAM2 prompt creation, propagation, and masks.

## Segmental implementation

`segmental.py` implements log-space forward and per-terminal backward recursions.
The learned termination logits form `P(N|q)`; the reported sequence score is the
log-sum-exp over all terminal prefixes and valid stay/advance-one paths. A safe
log-sum-exp explicitly handles unreachable states so backward gradients remain
finite.

The direct test suite covers:

- forward partition versus brute-force enumeration;
- variable-length marginalization versus brute force;
- posterior and terminal-posterior normalization;
- no backward transition and self-loop duration;
- finite gradients, permutation response, one-state behavior, and `M>T` error;
- training object-discrimination loss and differentiable mask-pooled object
  scores;
- deterministic GroundMoRe parser behavior;
- absence of GT fields in the inference capture hook.

The VIRST environment does not contain the `pytest` package, so these tests are
also executable as dependency-free direct Python test functions. No package was
installed into the shared environment.

## GT boundary

The inference wrapper captures only input IDs, attention masks, frozen hidden
states, and official visual features. Training-only object masks are loaded by a
separate module after the model forward and are used only for `L_obj`; they never
enter prompt creation or inference.

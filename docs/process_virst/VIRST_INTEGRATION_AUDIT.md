# VIRST integration audit

Audited source: official VIRST commit
`00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`.

## Verified information flow

1. `VirstMetaModel.initialize_module` creates the official SAM2 executor,
   `SegPrompter`, occurrence head, and `InitialSegFusion`.
2. Before the language-model forward, `InitialSegFusion` downsamples SAM2 video
   features and cross-attends the input `[SEG]` embedding to those spatiotemporal
   features. Its residual gate is part of the official model.
3. The language model produces final hidden states. VIRST extracts the final-layer
   hidden state at every `[SEG]` token.
4. `SegPrompter` projects each `[SEG]` state to 256 dimensions, repeats it over
   frames, and runs two RoPE-aware decoder layers against downsampled per-frame
   SAM2 feature tokens. Its output is a frame-specific prompt sequence and its
   cross-attention yields frame scores.
5. Evaluation selects uniformly spaced conditioning frames. Training selects
   random conditioning frames plus nearby propagation frames.
6. The selected frame-specific prompt vectors are passed to official SAM2 either
   through the batched training forward or `add_new_prompt`, followed by official
   video propagation.

Therefore VIRST already makes temporal visual information available before pixel
execution. The ProcessVIRST insertion point is the repeated 256-dimensional
per-frame segmentation state **inside `SegPrompter`, before its decoder and before
any SAM2 prompt or mask exists**.

## ProcessVIRST overlay

The overlay in `projects/evoseg/process_virst/` does not modify or vendor official
VIRST. It:

- captures frozen query-token hidden states after the official language backbone;
- builds four ordinal process slots by cross-attending those query states;
- cross-attends each slot to every frame's official SAM2 spatial features;
- applies a differentiable, strictly monotonic, validity-gated log-sum-exp DP;
- forms a process-conditioned residual for every frame;
- applies `LayerNorm(h_t + beta * Adapter(c_t))`, with positive `beta=1.0` at
  initialization;
- then runs the unmodified official `SegPrompter` spatial projection and decoder.

The official SegPrompter weights, frame-selection policy, SAM2 prompt API, and
propagation behavior remain unchanged. The wrapper exposes original and fixed
permutation alignment scores for SFT order loss and diagnostics.

## Stable official components versus research overlay

| Component | Status |
|---|---|
| SAM2 image encoder / memory / propagation | Official, frozen |
| InitialSegFusion | Official checkpoint behavior |
| VLM backbone | Official, frozen |
| SegPrompter decoder | Official weights; train policy recorded per run |
| Process Compiler / monotonic DP / adapter / beta | New research module |
| Reverse and half-block permutation | Training diagnostic only; never new GT |

No mask-track post-processing, SAM3.1 candidate bank, point/box reconstruction, or
post-hoc track selector is present in this path.

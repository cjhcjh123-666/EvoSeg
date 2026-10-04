# Factorized Temporal Grounding implementation

Qwen3-VL encodes the ordered sampled frames and original referring expression in
one multimodal sequence. Exact visual-token spans are recovered from
`image_grid_thw`; mean pooling within each span gives frame states `H_t`, while
the expression token span gives query state `q`.

The grounding interface in `projects/evoseg/ftg/interface.py` implements all six
controlled variants. FTG first obtains one identity state from the query and the
mean video context. Each frame state is then transformed while conditioned on
that same identity. Separate projections map identity and state to SAM3.1's
256-dimensional prompt space, and a vector gate controls the dynamic residual.
The final dynamic-state projection is zero-initialized, so FTG begins exactly at
the persistent Identity Only solution and learns temporal corrections from mask
supervision instead of injecting a random state perturbation at initialization.

The dynamic prompt enters official SAM3.1 through
`Sam3Image._encode_prompt(..., visual_prompt_embed=...)`. The official frozen
vision-language encoder, object-query decoder, and segmentation head produce the
candidate mask logits and object-query features. FTG uses the persistent identity
representation for a second, structurally distinct role: a lightweight learned
alignment head scores each candidate object query against the same identity in
every frame. The aligned score is added to SAM3.1's native objectness. Thus
identity controls *which object persists*, while the state residual controls
*how that object is segmented now*. The Frame Prompt control uses the identical
alignment head but supplies its frame-dependent monolithic representation, so
the contrast isolates persistent identity rather than extra supervision or
parameters.

During training, one-target matching supervises mask BCE, Dice, and the composed
query score. At inference, that same learned grounding score selects the query;
ground truth never constructs a prompt or selects an output. This association
head is part of the end-to-end grounding interface, not a post-hoc verifier.

`--query-score-mode native` retains SAM3.1's predicted query score as the
diagnostic baseline; `--query-score-mode representation` activates identity-
aware query association. A controlled
`fixed_slot` interface is also available for the diagnosed permutation problem:
one predetermined official SAM query receives mask supervision and the same
query is read at inference. It never searches queries with ground truth and adds
no scorer, verifier, or post-hoc selection module.

Trainable parameters are Qwen LoRA in the final six language layers plus only the
active controlled grounding branch. All SAM3.1 parameters, the old foundation
fusion module, inactive controls, and the old bridge remain frozen.

## Entry points

- `python -m projects.evoseg.ftg.train_pilot`: train/evaluate one variant.
- `python -m projects.evoseg.ftg.summarize_pilot`: compare completed FTG and
  Frame Prompt runs against the preregistered pilot gate.
- `python -m projects.evoseg.ftg.evaluate_checkpoint`: separate actual
  SAM3.1 query-selected masks from training-only matched-query masks and run
  temporal-order diagnostics on a saved lightweight checkpoint.
- `projects/evoseg/ftg/public_video_data.py`: deterministic public-data manifest
  and loaders.
- `projects/evoseg/ftg/metrics.py`: region J, boundary F, J&F, and absent-frame
  error rates.

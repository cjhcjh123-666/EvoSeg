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

The resulting prompt enters official SAM3.1 through
`Sam3Image._encode_prompt(..., visual_prompt_embed=...)`. The official frozen
vision-language encoder, object-query decoder, and segmentation head produce the
mask logits. During training, one-target matching supervises mask BCE, Dice, and
the official query score. At inference, only SAM3.1's predicted score selects the
object query. Ground truth never constructs a prompt or selects an output.

Trainable parameters are Qwen LoRA in the final six language layers plus only the
active controlled grounding branch. All SAM3.1 parameters, the old foundation
fusion module, inactive controls, and the old bridge remain frozen.

## Entry points

- `python -m projects.evoseg.ftg.train_pilot`: train/evaluate one variant.
- `python -m projects.evoseg.ftg.summarize_pilot`: compare completed FTG and
  Frame Prompt runs against the preregistered pilot gate.
- `projects/evoseg/ftg/public_video_data.py`: deterministic public-data manifest
  and loaders.
- `projects/evoseg/ftg/metrics.py`: region J, boundary F, J&F, and absent-frame
  error rates.

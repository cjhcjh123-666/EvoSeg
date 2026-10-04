# Factorized Temporal Grounding implementation

## Active strong-foundation design

The primary implementation starts from the public
`Sa2VA-Qwen3-VL-4B-SAM3` checkpoint. It already maps Qwen's generated `[SEG]`
state through a learned projection and uses the result to condition the native
SAM3 video tracker. Its released inference path repeats one monolithic language
embedding on every frame before propagation. FTG preserves these pretrained
weights but assigns two representations to different responsibilities:

- `z_id` is computed once from the query and global video context. It owns the
  tracker object ID and remains attached to persistent SAM memory.
- `z_state_t` is conditioned on `z_id` and the current Qwen frame state. It
  supplies a frame-dependent observation that may recondition the current mask
  and memory without changing object ownership.

This makes the interface recurrent rather than additive. Identity is not
recomputed independently per frame, and state is not added to identity to form
one prompt vector. The frozen native SAM tracker carries the persistent state;
Qwen LoRA and the identity/state observation projections are trainable. Ground
truth supervises masks during training but never initializes memory, chooses an
anchor, or selects a prediction at evaluation time.

The decisive Frame Prompt control receives the same frame observations and
trainable budget but has no persistent identity variable. Identity Memory uses
the persistent representation without frame reconditioning. State Only restarts
from frame-dependent evidence without carrying an identity. The old vector-sum
variant is retained as a negative architectural control.

## Rejected scratch implementation

The existing `projects/evoseg/ftg/interface.py` and `train_pilot.py` implement
the first scratch Qwen3-VL-4B + frozen official SAM3.1 study. They recover exact
visual-token spans from `image_grid_thw`, form frame states `H_t`, and evaluate
Global, Frame, Identity, State, and additive FTG prompts through SAM3.1's
`visual_prompt_embed` path. SAM parameters remain frozen and query-selected
predictions never use ground truth.

That implementation is useful as a controlled foundation ablation, but it is
not the active paper method. Across the scaled 192-expression, three-seed pilot,
Frame Prompt reached 43.45 +/- 0.31 J&F and additive FTG reached 39.62 +/- 2.25.
A stronger training-free test that wrote the complete Qwen anchor mask through
SAM3.1's native memory encoder reached only 21.25 J&F. These results reject both
prompt-vector addition and one-shot anchor propagation.

## Current entry points

- `projects/sa2va/evaluation/sa2va_eval_ref_vos.py`: reproducible strong
  Qwen3-VL/SAM3 foundation evaluation with explicit public-dataset path
  overrides and deterministic smoke subsets.
- `python -m projects.evoseg.ftg.evaluate_sam31_video_baseline`: frozen official
  SAM3.1 detector-to-tracker diagnostic.
- `python -m projects.evoseg.ftg.evaluate_qwen_seeded_tracker`: rejected
  training-free Qwen-mask-to-tracker-memory control.
- `python -m projects.evoseg.ftg.train_pilot`: legacy scratch controlled pilot.
- `projects/evoseg/ftg/public_video_data.py`: deterministic Long-RVOS and
  MeViS-v2 pilot manifests and loaders.
- `projects/evoseg/ftg/metrics.py`: region J, boundary F, J&F, present-frame
  J&F, and absent-frame error rates.

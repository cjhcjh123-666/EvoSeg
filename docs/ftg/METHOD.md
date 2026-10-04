# Factorized Temporal Grounding implementation

## Active strong-foundation design

The primary implementation starts from the public
`Sa2VA-Qwen3-VL-4B-SAM3` checkpoint. It already maps Qwen's generated `[SEG]`
state through a learned projection and uses the result to condition the native
SAM3 video tracker. Its released inference path repeats one monolithic language
embedding on every frame before propagation. FTG preserves these pretrained
weights but assigns two representations to different responsibilities:

- `z_id` is the checkpoint's pretrained `[SEG]` projection, computed from the
  query and video context. It is the persistent identity anchor.
- `z_state_t` is produced by using `z_id` as a single cross-attention query over
  the current frame's native SAM3 spatial features. It is therefore both
  target-conditioned and frame-dependent.

For frame `t`, the prompt is
`p_t = z_id + sigmoid(g_t) * residual(z_id, z_state_t)`. The residual's last
projection is zero-initialized, making the initial model exactly equal to the
public foundation rather than a randomly perturbed prompt. Qwen, the existing
`[SEG]` projection, and SAM3 are frozen; only 592,897 parameters in the state
branch and gate are optimized in the controlled pilot. Ground truth supervises
masks during training but never initializes memory, chooses an anchor, or
selects a prediction at evaluation time.

The decisive Frame Prompt control receives the same frame observations and
trainable budget but has no persistent identity variable. Identity Memory uses
the persistent representation without frame reconditioning. State Only restarts
from frame-dependent evidence without carrying an identity. The earlier broad
two-token FTG and random prompt-replacement variants are negative architectural
controls.

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
- `projects/sa2va/configs/ftg/ftg_qwen3_4b_sam3_video_pilot.py`: public
  MeViS-v2 + Long-RVOS broad-adaptation negative control.
- `projects/sa2va/configs/ftg/anchored_ftg_qwen3_4b_sam3_video_pilot.py`:
  foundation-preserving primary FTG pilot.
- `projects/evoseg/ftg/strong_interface.py`: identity-query spatial
  cross-attention and controlled prompt variants.
- `projects/evoseg/eval/eval_mevis_jf.py`: official native-resolution MeViS
  J/F evaluation.
- `projects/evoseg/eval/eval_long_rvos.py`: official Long-RVOS J/F, tIoU, and
  vIoU with Static/Dynamic/Hybrid reporting.
- `python -m projects.evoseg.ftg.evaluate_sam31_video_baseline`: frozen official
  SAM3.1 detector-to-tracker diagnostic.
- `python -m projects.evoseg.ftg.evaluate_qwen_seeded_tracker`: rejected
  training-free Qwen-mask-to-tracker-memory control.
- `python -m projects.evoseg.ftg.train_pilot`: legacy scratch controlled pilot.
- `projects/evoseg/ftg/public_video_data.py`: deterministic Long-RVOS and
  MeViS-v2 pilot manifests and loaders.
- `projects/evoseg/ftg/metrics.py`: region J, boundary F, J&F, present-frame
  J&F, and absent-frame error rates.

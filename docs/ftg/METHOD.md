# Factorized Temporal Grounding implementation

> Historical implementation, not the current training route. See the
> [user-approved 2026-10-06 restart](../EVOSEG_CURRENT.md).

## Active strong-foundation design

The primary implementation now starts from the public VIRST checkpoint. VIRST
already has a trained language-to-mask geometry and emits frame-specific
SegPrompter outputs `x_t`. FTG changes only the representation interface after
that released prompter:

\[
z^{id}=\frac{1}{T}\sum_t x_t,\qquad z_t^{state}=x_t-z^{id}.
\]

The persistent identity is therefore one video-level temporal mean and the
state is zero-mean by construction. An identity-conditioned channel gate makes
a bounded correction to the released state:

\[
\delta_t=\rho\tanh G([z^{id};z_t^{state}]),\qquad
p_t=x_t+\operatorname{center}_t(\delta_t\odot z_t^{state}).
\]

The last gate projection is initialized to zero, so `delta_t=0` and `p_t=x_t`
bit-for-bit before training. Re-centering the correction guarantees that the
temporal mean of `p_t` remains the fixed `z_id` after optimization. The public
VideoChat/VIRST/SAM2.1 weights and keyframe scores remain frozen; only the small
composer is trained with public segmentation masks. Ground truth never
initializes memory, selects prompts, or chooses predictions at inference.

Controls are the unmodified public VIRST prompt, Identity Only (repeat the
temporal mean), State Only (remove the persistent mean), exact factorization
without a learned gate, an unconditioned state gate, a scalar gate, and the
channel-gated FTG interface. This directly tests factorization while retaining
a strong step-zero baseline.

The Qwen3-VL-4B + SAM3 study remains a controlled negative result. Uniform
full-video evidence selectively helped Dynamic Long-RVOS queries but
destabilized Static/Hybrid identity. Its dual-source First-5 identity plus
Uniform-5 state interface fell from 60.057 to 49.559 J&F, so it is not promoted
to full training.

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
- `projects/sa2va/configs/ftg/joint_adapted_vector_gated_centered_ftg_qwen3_4b_sam3_video_pilot.py`:
  current centered, channel-gated FTG interface pilot.
- `projects/evoseg/ftg/strong_interface.py`: identity-query spatial
  cross-attention and controlled Qwen/SAM3 prompt variants.
- `projects/evoseg/ftg/virst_interface.py`: exact-initialized VIRST identity/state
  factorization and controlled prompt variants.
- `projects/evoseg/ftg/train_virst_ftg.py`: frozen-public-VIRST adapter training
  on public MeViS-v2 and Long-RVOS data.
- `projects/evoseg/ftg/virst_ftg_eval.py`: non-invasive injection into the
  official VIRST evaluator.
- `projects/evoseg/eval/eval_mevis_jf.py`: official native-resolution MeViS
  J/F evaluation.
- `projects/evoseg/eval/eval_long_rvos.py`: official Long-RVOS J/F, tIoU, and
  vIoU with Static/Dynamic/Hybrid reporting.
- `projects/evoseg/eval/compare_long_rvos.py`: paired expression deltas with
  source-video cluster-bootstrap confidence intervals.
- `projects/evoseg/ftg/visualize_long_rvos_comparison.py`: metric-selected,
  paired Long-RVOS qualitative sheets for each expression type.
- `python -m projects.evoseg.ftg.evaluate_sam31_video_baseline`: frozen official
  SAM3.1 detector-to-tracker diagnostic.
- `python -m projects.evoseg.ftg.evaluate_qwen_seeded_tracker`: rejected
  training-free Qwen-mask-to-tracker-memory control.
- `python -m projects.evoseg.ftg.train_pilot`: legacy scratch controlled pilot.
- `projects/evoseg/ftg/public_video_data.py`: deterministic Long-RVOS and
  MeViS-v2 pilot manifests and loaders.
- `projects/evoseg/ftg/metrics.py`: region J, boundary F, J&F, present-frame
  J&F, and absent-frame error rates.

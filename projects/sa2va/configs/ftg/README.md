# FTG training configs

These Qwen3-VL/SAM3 configs are retained as controlled foundation ablations.
The Long-RVOS dual-source gate showed a significant overall regression, so the
paper-scale method now uses the exact-initialized public VIRST interface in
`projects/evoseg/ftg/virst_interface.py` and
`projects/evoseg/ftg/train_virst_ftg.py`.

`ftg_qwen3_4b_sam3_video_pilot.py` is the first temporal-only gate. It uses
only the public MeViS-v2 and Long-RVOS train splits and explicitly disables the
historical existence/faithfulness head. The initialization checkpoint is a
lossless training-format conversion of the public local
`Sa2VA-Qwen3-VL-4B-SAM3` Hugging Face export.

The loader normalizes legacy SAM3 HF `g_weight` layer-scale keys to the
training model's `gamma` spelling. This compatibility step is required for the
two mask-memory fuser blocks: silently missing them changes video propagation
even when SAM3 is frozen. A valid export must reproduce the public checkpoint
under `identity_memory` before a learned interface is scored.

The pilot keeps SAM3 frozen so the controlled comparison changes only Qwen LoRA,
the pretrained `[SEG]` projection, and the FTG state module. After the temporal
gate, the full Public-SegMix run adds RefCOCO/+/g and Ref-Youtube-VOS without
changing the model or optimization recipe.

`identity_adapt_qwen3_4b_sam3_video_pilot.py` is the clean stage-one baseline.
It uses the same public video data, Qwen LoRA, projection, and optimization
budget as joint FTG while retaining the published single identity prompt. It
must be compared directly with joint FTG and Frame Prompt, and its checkpoint
is the intended foundation for the two-stage adapted Anchored FTG run.

`frame_prompt_qwen3_4b_sam3_video_pilot.py` is the broad-adaptation Frame Prompt
control. It inherits the same checkpoint, data order, and training budget as
the first two-token FTG run, but emits only the frame-dependent state token. It
is not parameter-matched to the later frozen-foundation recipe.

`anchored_ftg_qwen3_4b_sam3_video_pilot.py` is the foundation-preserving
follow-up after the first full FTG run regressed on MeViS. It freezes Qwen and
the pretrained `[SEG]` projection, disables LoRA, and learns only a
zero-initialized, identity-conditioned dynamic residual. Its step-zero prompt
is exactly the original single identity token; this prevents broad fine-tuning
or a random extra sparse token from destroying the reproduced strong baseline.

`unconditioned_residual_qwen3_4b_sam3_video_pilot.py` is the parameter-matched
control for Anchored FTG. It uses the same frozen foundation, 592,897 trainable
parameters, zero initialization, data, and schedule, but replaces the identity
query used for state extraction with zeros. Anchored FTG versus this control
isolates identity-conditioned dynamic state from a generic frame residual.

`bounded_ftg_qwen3_4b_sam3_video_pilot.py` adds a functional trust region after
the unbounded Anchored FTG prompt was found to remap instances despite small
weight norms. The dynamic residual may move the frozen identity prompt by at
most 2% of its norm. This is a hard prompt-space bound, not a weight-decay
proxy; step zero still exactly reproduces the public foundation.

`adapted_anchored_ftg_qwen3_4b_sam3_video_pilot.py` is the clean two-stage
follow-up. It uses the public-video-adapted identity checkpoint as its frozen
foundation, explicitly ignores that checkpoint's jointly trained
`factorized_grounding` weights, and learns a fresh zero-initialized state
residual. This makes the decisive comparison adapted Identity versus adapted
Identity + State without moving Qwen and the interface simultaneously.

`joint_adapted_bounded_ftg_qwen3_4b_sam3_video_pilot.py` instead retains the
stronger identity representation discovered inside the corrected joint FTG
checkpoint, discards its harmful two-token state module, and learns a new
single-token residual within a 2% identity-prompt trust region.

`joint_adapted_centered_ftg_qwen3_4b_sam3_video_pilot.py` is the temporal-state
definition test. It initializes from the small bounded-residual checkpoint,
removes each object's across-frame mean observation, and retrains only that
factorized module. The overlay is loaded before DeepSpeed wrapping and is
restricted to `factorized_grounding.*` keys. This stage uses no faithful,
no-target, synthetic, or generated training data.

`joint_adapted_gated_centered_ftg_qwen3_4b_sam3_video_pilot.py` corrects the
fixed-budget centered control so the learned gate remains functional after
state direction normalization. It additionally centers the produced state
over time and bounds each frame's residual to
`gate_t * 0.02 * ||identity||`. It initializes from the trained centered stage
and otherwise preserves the same frozen foundation, public data, and budget.

`joint_adapted_vector_gated_centered_ftg_qwen3_4b_sam3_video_pilot.py` is the
final interface pilot before the benchmark matrix. The scalar gate was nearly
constant within each expression, so this control replaces it with the
factorized per-channel gate in the paper equation, `g_t \odot state_t`. The
gate is applied after state normalization, the complete dynamic correction is
still hard-bounded to 2% of the identity-prompt norm, and only the small FTG
module is updated from the scalar-gated checkpoint.

# FTG training configs

`ftg_qwen3_4b_sam3_video_pilot.py` is the first temporal-only gate. It uses
only the public MeViS-v2 and Long-RVOS train splits and explicitly disables the
historical existence/faithfulness head. The initialization checkpoint is a
lossless training-format conversion of the public local
`Sa2VA-Qwen3-VL-4B-SAM3` Hugging Face export.

The pilot keeps SAM3 frozen so the controlled comparison changes only Qwen LoRA,
the pretrained `[SEG]` projection, and the FTG state module. After the temporal
gate, the full Public-SegMix run adds RefCOCO/+/g and Ref-Youtube-VOS without
changing the model or optimization recipe.

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

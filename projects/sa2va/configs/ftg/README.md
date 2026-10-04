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

`frame_prompt_qwen3_4b_sam3_video_pilot.py` is the decisive controlled baseline.
It inherits the exact checkpoint, seed, data order, optimization, and training
budget, but sends only the frame-dependent state token to SAM3. The difference
from FTG is therefore the persistent identity token, not frame conditioning.

`anchored_ftg_qwen3_4b_sam3_video_pilot.py` is the foundation-preserving
follow-up after the first full FTG run regressed on MeViS. It freezes Qwen and
the pretrained `[SEG]` projection, disables LoRA, and learns only a
zero-initialized, identity-conditioned dynamic residual. Its step-zero prompt
is exactly the original single identity token; this prevents broad fine-tuning
or a random extra sparse token from destroying the reproduced strong baseline.

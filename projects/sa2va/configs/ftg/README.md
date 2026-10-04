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

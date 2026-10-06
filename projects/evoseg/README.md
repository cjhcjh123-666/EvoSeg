# projects/evoseg — EvoSeg research layer

Own additions on top of the Pixel-LLM/Sa2VA base.

The active public-benchmark segmentation hallucination project is documented in
[EVOSEG_CURRENT.md](../../docs/EVOSEG_CURRENT.md).
Use existing public training data and benchmarks only. No new benchmark or
generated dialogue dataset is part of this project. The first matched
baseline pair is released Sa2VA-Qwen3-VL-4B and the saved Faithful-4B with
their original SAM2. New training starts independently from public
Qwen3-VL-8B-SAMTok + native SAM2.1, using original RefCOCO/+/g and gRefCOCO
TRAIN only. Old Faithful weights and dialog LoRAs are not inherited.
No SOTA claim is made before full comparable evaluation.
Do not resume multi-turn/FTG/RVOS-restart jobs or unvalidated old video SFT.

Key directories:

- `hallucination/`: active public protocols, baselines and grounding-interface hypothesis.
- `dialog/`: historical multi-turn project, checkpoints and unreviewed data drafts.
- `restart/`: completed RVOS foundation, repair and negative-result provenance.
- `ftg/`: historical factorized grounding experiments and diagnostics, not active.
- `qwen_process_seg/`: historical audited Qwen3-VL/SAM3.1 foundation plumbing.
- earlier faithfulness/process directories: read-only research provenance;
  only audited public metrics and unchanged checkpoints may enter new controls.

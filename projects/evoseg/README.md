# projects/evoseg — EvoSeg research layer

Own additions on top of the Pixel-LLM/Sa2VA base.

The active multi-turn segmentation project is documented in
[EVOSEG_CURRENT.md](../../docs/EVOSEG_CURRENT.md).
Start from released Qwen3-VL-8B-SAMTok with its native pixel foundation.
Public MRSeg experiments and a reviewed, public-GT dialogue-data pilot are
separate tracks. No SOTA claim is made before complete comparable evaluation.
Do not resume the completed RVOS/FTG/faithfulness runs.

Key directories:

- `dialog/`: current scope-controlled multi-turn project and data preparation.
- `restart/`: completed RVOS foundation, repair and negative-result provenance.
- `ftg/`: historical factorized grounding experiments and diagnostics, not active.
- `qwen_process_seg/`: historical audited Qwen3-VL/SAM3.1 foundation plumbing.
- earlier faithfulness/process directories: retained for provenance, not active in
  the FTG paper story.

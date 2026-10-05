# projects/evoseg — EvoSeg research layer

Own additions on top of the Pixel-LLM/Sa2VA base.

The active restart is documented in [EVOSEG_CURRENT.md](../../docs/EVOSEG_CURRENT.md).
Start from the released strong segmentation VLM with its original SAM and
inference protocol; the first candidate is SaSaSa2VA-26B with native SAM2.
Do not resume the previous FTG primary job or replace its pixel foundation now.

Key directories:

- `restart/`: pinned strong-foundation preparation for the new execution chain.
- `ftg/`: historical factorized grounding experiments and diagnostics, not active.
- `qwen_process_seg/`: historical audited Qwen3-VL/SAM3.1 foundation plumbing.
- earlier faithfulness/process directories: retained for provenance, not active in
  the FTG paper story.

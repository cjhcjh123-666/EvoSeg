# projects/evoseg — EvoSeg research layer

Own additions on top of the Pixel-LLM/Sa2VA base.

The active method is `ftg/`: Qwen3-VL frame/query states are factorized into one
persistent identity representation and dynamic state representations, then sent
through the audited official SAM3.1 `visual_prompt_embed` path.

Key directories:

- `ftg/`: active factorized grounding model, public pilot data, training, metrics,
  and controlled tests.
- `qwen_process_seg/`: audited Qwen3-VL/SAM3.1 foundation plumbing reused by FTG.
- earlier faithfulness/process directories: retained for provenance, not active in
  the FTG paper story.

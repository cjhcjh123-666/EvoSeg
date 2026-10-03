# Foundation data protocol

- Qwen input budget: 16 uniformly sampled frames over the complete source video; fewer than 16 available frames are all used once.
- Every record stores source frame indices, timestamps when available, `image_grid_thw`, frame token spans, and total visual-token count.
- The historical capability audit used official Long-RVOS train examples only.
  The active FTG pilot uses public Long-RVOS train plus MeViS-v2 train under the
  fixed manifest in `docs/ftg/PILOT_PROTOCOL.md`.
- Evaluation objects, expressions, masks, and bootstrap grouping reuse the established official protocols. Validation/test examples never train the bridge, LoRA, or process modules.
- No LLM classification, rewritten query, artificial event label, synthetic
  expression, or pseudo mask is permitted.
- SAM3.1 visual inputs and Qwen inputs use their fixed native preprocessing. Memory pressure may reduce worker concurrency, never frame count, prompt count, or resolution.

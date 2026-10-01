# Data protocol

- Qwen input budget: 16 uniformly sampled frames over the complete source video; fewer than 16 available frames are all used once.
- Every record stores source frame indices, timestamps when available, `image_grid_thw`, frame token spans, and total visual-token count.
- Training: official Long-RVOS train and GroundMoRe Sequential trainval only. MeViS and Ref-YT-VOS are deferred until the core method gate.
- Evaluation objects, expressions, masks, and bootstrap grouping reuse the established official protocols. Validation/test examples never train the bridge, LoRA, or process modules.
- Long-RVOS order-sensitive examples use only the preregistered deterministic connective filter. No LLM classification, rewritten query, artificial event label, or pseudo mask is permitted.
- Reverse and half-block-swap sequences change only process-reasoner input and never receive the original segmentation mask as reversed-video supervision.
- SAM3.1 visual inputs and Qwen inputs use their fixed native preprocessing. Memory pressure may reduce worker concurrency, never frame count, prompt count, or resolution.

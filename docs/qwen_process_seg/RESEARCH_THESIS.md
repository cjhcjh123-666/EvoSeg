# QwenProcessSeg research thesis

QwenProcessSeg tests one claim: a video-language model should form a continuous, query-conditioned object-process representation **before** SAM3.1 predicts or tracks pixels. Qwen3-VL-4B supplies language and frame-aware video states; a continuous process reasoner supplies per-frame grounding states; a small bridge translates those states into the frozen SAM3.1 prompt space. SAM3.1 remains the only pixel decoder and tracker.

This branch excludes the previous frozen-candidate and post-hoc selection routes. It does not initialize from Faithful, VideoFaithful, TEG, verifier, refusal, or rejection checkpoints. The base model is the unmodified local `Qwen3-VL-4B-Instruct` checkpoint.

The gated execution order is: official SAM3.1 prompt-gradient audit, frame-aware QwenSeg-SAM31 capability, process-reasoner capability, pilot, then full SFT. RL is out of scope until a full SFT gate passes.

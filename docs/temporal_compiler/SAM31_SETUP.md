# SAM 3.1 setup and capability audit

Audit date: 2026-09-22 (Asia/Shanghai)

## Provenance and runtime

| Item | Verified value |
|---|---|
| Official repository | `https://github.com/facebookresearch/sam3` |
| Repository commit | `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| Release audited | SAM 3.1, dated 2026-03-27 in `RELEASE_SAM3p1.md` |
| Runtime builder | `sam3/model_builder.py::build_sam3_multiplex_video_predictor` |
| Checkpoint | `sam3.1_multiplex.pt`, 3,502,755,717 bytes |
| Checkpoint SHA-256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| Download source used | Domestic Hugging Face mirror `AEmotionStudio/sam3.1-byte-identical-official-sha` |
| Checkpoint verification | Byte hash equals the expected official SAM 3.1 checkpoint hash; no result was selected by model output |
| Isolated environment | `/9950backfile/chenjiahui/evo_artifacts/envs/sam31` |
| Python / PyTorch / CUDA | Python 3.12.14 / PyTorch 2.7.1+cu118 / CUDA runtime 11.8 |
| Hardware tested | NVIDIA A800-SXM4-80GB, compute capability 8.0 |

The Meta repository is clean at the audited commit. It was not patched. Project-side compatibility code lives in `projects/evoseg/temporal_compiler/sam31_candidate_protocol.py`.

The final load audit compared the checkpoint with the instantiated official model: 1,623 checkpoint keys, 1,687 model keys, no unexpected keys, and no disallowed missing keys. The 64 missing tensors are deterministic real/imaginary RoPE buffers rebuilt by the official `use_rope_real=True` path. Three tensors from the language encoder, geometry encoder, and tracker were additionally compared value-for-value after loading.

## Official capabilities checked in code

The release notes and official example expose Object Multiplex as a joint multi-object tracker using fixed-capacity buckets. The recommended builder accepts `max_num_objects`, `multiplex_count`, FlashAttention, real-valued RoPE, and compile controls. Both the base and multiplex predictors use the same session API:

1. `start_session` with a video resource path.
2. `add_prompt` with a frame index and text (or supported geometric prompt).
3. `propagate_in_video` as a streamed request.
4. `remove_object`, `reset_session`, and `close_session` for session management.

For a text prompt, the official example states that all matching instances are detected, assigned distinct object IDs, and tracked through the video. Streamed outputs used in the audit contain `out_obj_ids`, `out_binary_masks`, and `out_probs`. Those fields are sufficient to construct a candidate track bank with per-track masks and an aggregate confidence. They do not by themselves solve which track satisfies a relational Dynamic expression; that is the capability being measured, not assumed.

The audited builder uses image size 1008, text-conditioned detection, multi-object tracking, a default output mask threshold of 0.5, detection threshold 0.4, `max_num_objects=16`, and `multiplex_count=16` in our run. FlashAttention 3 and `torch.compile` were disabled for the correctness pilot because the installed A800/CUDA stack did not provide the release's optimized H100 path. Resolution and thresholds are unchanged across prompt conditions.

## Compatibility finding

At the audited commit, the public predictor wrapper still forwards CPU-offload keyword arguments used by the older session wrapper, while the multiplex model's inspected `init_state` signature no longer accepts them. The project adapter therefore calls the inspected official model signature and omits only unsupported CPU-offload keywords. It does not change detection, tracking, prompt, or mask logic. A real end-to-end smoke subsequently completed, so this is classified as an official wrapper/API mismatch rather than a checkpoint failure.

## GT-separation and candidate protocol

Candidate generation reads only video frames and one of the following prompts:

- the unchanged official expression;
- a deterministic noun/noun-phrase parse of that expression;
- a separately logged Qwen text-only parse of that expression.

It never reads the Long-RVOS target ID or mask. Candidate IDs, RLE masks, confidence observations, prompt method, and frame names are saved first. A separate evaluation command later reads GT to choose the oracle-best candidate and compute J, F, J&F, recall thresholds, and candidate-count statistics. Duplicate successful keys across shards are rejected.

## Real inference status

A one-object smoke completed 27/27 prompt-expression records with zero failures. All three prompt methods were run on the object's three Static, three Dynamic, and three Hybrid expressions. This is an interface and data-flow check, not a coverage conclusion.

For that one object only, concept prompts achieved oracle J&F 0.9302/0.9348/0.8100 for Static/Dynamic/Hybrid, and the Qwen concepts achieved 0.9348 for all three types. Raw expressions achieved 0.8287/0.3090/0.2161. Recall@0.5 was 1.0 for concept and Qwen conditions, while raw Dynamic and Hybrid were 0.3333. These values remain explicitly preliminary because the sample size is one object.

The prediction-independent 32-object pilot is running as four object-level shards on GPUs 0--3. Final candidate coverage and the go/no-go decision remain pending the merged object-weighted evaluation; no method route is inferred from the smoke.

## Suitability verdict at setup stage

SAM 3.1 is technically suitable for testing as a candidate-track / pixel executor: the official checkpoint loads, text prompting produces multiple persistent object IDs and mask tracks, and candidate generation can be isolated from GT. Whether it is scientifically suitable for the proposed matcher is not yet established. That requires high target coverage on the 32-object pilot and then the full paired-object protocol; a low-coverage outcome is an explicit no-go for matcher training.

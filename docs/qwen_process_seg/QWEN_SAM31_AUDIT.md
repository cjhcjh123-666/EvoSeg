# Qwen3-VL and SAM3.1 infrastructure audit

## Provenance

- Git base: `origin/main@f1a39a105d35670d8b7845fda38d213e0b4c0664`.
- Research branch: `research/qwen-process-seg-vlm`.
- Qwen initialization: `/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-4B-Instruct`; 36 language layers, hidden size 2560, 24-layer vision encoder, visual output size 2560.
- Official SAM checkout: `/9950backfile/chenjiahui/evo_artifacts/external/sam3@2345a4ad109ac29c569da749c91d84f10dc08c40`.
- Official SAM3.1 checkpoint: `sam3.1_multiplex.pt`, SHA256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.

## A. Qwen3-VL video path

The existing training dataset defaults to five sampled frames (`Sa2VA03RefVOS.sampled_frames=5`). The HF video inference wrapper is more restrictive: it appends only frames with `frame_idx < 5` to Qwen's content while preprocessing every frame for the grounding model. Therefore the existing path does **not** satisfy the 16-frame full-range protocol.

Qwen image/video tokens enter the language sequence as visual placeholder positions and are replaced by visual embeddings inside `Qwen3VLForConditionalGeneration`. The current dataset already records `image_grid_thw`, per-frame token count, and frame count, but it does not persist source frame indices or explicit hidden-state spans.

QwenProcessSeg encodes 16 uniformly sampled frames as 16 ordered image items. This deliberately preserves one `image_grid_thw` row and one contiguous `<|image_pad|>` run per frame. `frame_protocol.recover_frame_token_spans` validates the exact count and stores `[start,end)` spans plus source frame indices. The final Qwen hidden state can then provide both the full spatial token sequence `V_t` and its per-frame summary `f_t`; no all-frame mean pooling is used.

## B. SAM wrapper and release status

The repository's `third_parts/sam3` is a slim, pre-Multiplex tracker subset. It has no Object Multiplex implementation and therefore is not SAM3.1, despite newer config names containing `sam3`. The actual SAM3.1 implementation is the separate official checkout listed above. Meta's March 27, 2026 release introduces Object Multiplex and new checkpoints; the official code identifies `build_sam3_multiplex_video_predictor` as the recommended public predictor entry point. See [Meta SAM3 repository](https://github.com/facebookresearch/sam3) and [SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

The public predictor supports session-based video inference, text and geometric prompt handling, memory, propagation, identity management, and multiplexed objects. It does **not** expose an arbitrary learned language embedding in its public request API.

Internally, the official checkpointed `MultiplexMaskDecoder` exposes `extra_per_object_embeddings: [num_buckets,multiplex_count,256]`, which is added to per-object mask tokens before pixel decoding. This is the only verified differentiable learned object-conditioning input suitable for the bridge. It is an official internal model interface, not a stable public wrapper contract; QwenProcessSeg must isolate it behind a tested adapter.

## C. Existing Qwen-to-SAM route

The existing HF wrapper extracts the last-layer hidden state at generated `[SEG]`, maps it with `text_hidden_fcs`, and repeats the same embedding with `[seg_hidden_states] * num_frames`. Its Qwen video input uses only the first five frames. The training model similarly repeats each object embedding over every video frame in `generate_video_pred_embeddings`. It also contains an old temporal-existence gate. None of those paths is reused by QwenProcessSeg.

The required replacement is:

1. exact 16-frame Qwen token spans;
2. frame-specific Qwen states;
3. optional process-conditioned frame states before pixel decoding;
4. a representation-only 256-dimensional bridge;
5. one SAM3.1 object slot with a distinct embedding per anchor frame.

## D. Gradient path

The official visual backbone can run under `torch.no_grad`, while the frozen Multiplex mask decoder remains differentiable with respect to `extra_per_object_embeddings`. The real-checkpoint audit passed; full measurements are in `SAM31_GRADIENT_AUDIT.md`. This establishes the required mask-loss-to-bridge direction without training a custom segmentation head.

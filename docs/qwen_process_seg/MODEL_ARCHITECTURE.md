# Model architecture

## Baseline: QwenSeg-SAM31

Sixteen uniformly sampled, full-range video frames and the official query are encoded by frozen Qwen3-VL-4B plus LoRA on the last six language layers. Visual token runs are recovered exactly for each frame. A gated residual combines frame summary `f_t` with the full-query summary `q_global`, and a small `LN-Linear-GELU-Linear` bridge maps 2560 dimensions to the official SAM3.1 256-dimensional object-token space.

For each anchor frame, the bridge output is added as a residual to SAM3.1's checkpointed valid-object embedding in one stable multiplex object slot. Other slots retain the checkpointed invalid-object embedding. The frozen official Multiplex mask decoder produces all mask logits. There is no custom mask head.

## Process model

The process variant will add a language-anchored, prefix-halted process compiler and PRE→p1…pN→POST content-adaptive segmental reasoner between Qwen states and the same bridge. Its posterior yields per-frame process context and transition confidence before SAM3.1 decoding. This module is not authorized for training until the baseline capability gate passes.

## Execution

Dense-frame prompting injects all available 16 anchor embeddings using one object identity, then uses official SAM3.1 memory/propagation for the full evaluation range. The key-frame ablation deterministically selects four anchors from process transition posterior only after the process capability gate. Ground truth never selects anchors or enters inference.

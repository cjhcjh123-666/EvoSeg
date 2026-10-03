# Model architecture

## Baseline: QwenSeg-SAM31

Sixteen uniformly sampled, full-range video frames and the official query are encoded by frozen Qwen3-VL-4B plus LoRA on the last six language layers. Visual token runs are recovered exactly for each frame. A gated residual combines frame summary `f_t` with the full-query summary `q_global`, and a small `LN-Linear-GELU-Linear` bridge maps 2560 dimensions to the official SAM3.1 256-dimensional object-token space.

For each anchor frame, the bridge output enters the official SAM3.1 image-grounding encoder through its learned `visual_prompt_embed` token input. The frozen official vision-language encoder, object-query decoder, and segmentation head produce mask logits. During SFT, standard one-target object-query assignment supplies mask BCE+Dice and query-ranking supervision; at inference, SAM3.1's own predicted grounding score selects the query. GT never constructs a prompt or selects an inference result. There is no custom mask head.

## Active extension

FTG replaces the monolithic fusion/bridge with the controlled identity-state
grounding interfaces documented in `docs/ftg/METHOD.md`. The process compiler and
segmental reasoner described in earlier drafts are retired and are not trained.

## Execution

The controlled pilot injects one prompt for each of 16 sampled frames through the
official image-grounding path. Ground truth never selects an inference query or
enters a prompt.

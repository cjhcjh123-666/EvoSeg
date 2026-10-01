# SAM3.1 prompt-gradient audit

## Result: PASS

The audit loaded the complete official SAM3.1 Object Multiplex model and verified its assembled checkpoint without reported final missing or unexpected keys. A deterministic synthetic 1008×1008 image was passed through the real frozen tri-head visual backbone under `no_grad`. Its propagation features were then decoded by the real frozen Multiplex mask decoder with an injected `extra_per_object_embeddings` tensor requiring gradients.

| Measurement | Result |
|---|---:|
| checkpoint SHA256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| prompt interface | `MultiplexMaskDecoder.extra_per_object_embeddings` |
| prompt shape | `[1,16,256]` |
| visual feature | `[1,256,72,72]` |
| high-resolution features | `[1,32,288,288]`, `[1,64,144,144]` |
| mask-logit output | `[1,288,288]` (object slot 0, one mask token) |
| BCE loss | 0.586230 |
| `||grad(z)||` | **0.084681** |
| maximum absolute prompt gradient | 0.012024 |
| gradient finite | yes |
| mask-logit MAE after `z += 0.1` | **0.074263** |
| frozen SAM3.1 parameters | 873,185,516 |
| trainable SAM3.1 parameters | 0 |
| SAM3.1 parameters receiving gradients | 0 |
| peak allocated GPU memory | 3.846 GiB |
| elapsed | 36.21 s |

The important distinction is that this is an official **internal object-token conditioning interface**, not the public text-prompt request API. It is nevertheless part of the official SAM3.1 decoder used by the checkpoint, and the audit proves that mask loss can update a Qwen bridge while every SAM3.1 weight remains frozen.

Raw result: `/9950backfile/chenjiahui/evo_artifacts/results/qwen_process_seg/20261001_qps_v1/audits/sam31_gradient_audit.json`.

The gate authorizes the frame-aware QwenSeg-SAM31 baseline. It does not yet establish tracking continuity or baseline segmentation quality; those remain separate gates.

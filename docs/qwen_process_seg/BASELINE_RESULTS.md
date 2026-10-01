# QwenSeg-SAM31 baseline gate

## Real two-sample smoke

The corrected QwenSeg-SAM31 path was exercised on two official Long-RVOS train expressions, with 16 uniformly sampled full-range frames per expression and four passes over the two records. This is a plumbing/capability smoke, not a validation result.

- Qwen: pure `Qwen3-VL-4B-Instruct`; no Sa2VA, Faithful, verifier, or old temporal checkpoint.
- LoRA: last six Qwen language layers, `q/k/v/o`, rank 16.
- Pixel output: frozen official SAM3.1 Multiplex decoder mask logits; no custom mask head.
- Loss: mask BCE + Dice on the official SAM3.1 output.
- Trainable parameters: LoRA 1,966,080; frame/query fusion 3,945,985; bridge 1,447,680.
- Final gradient norms: LoRA 0.1831, fusion 0.0844, bridge 1.9710.
- SAM3.1 trainable parameters / parameters receiving gradients: 0 / 0.
- Mean cross-frame prompt standard deviation: 0.0540.
- Empty-mask / full-mask fractions: 0 / 0.
- Peak allocated GPU memory: 14.69 GiB.
- Mean loss, first/final pass: 1.19744 / 1.19722.

The smoke therefore passes the gradient, frozen-executor, frame-specific prompt, and nondegenerate-mask checks. The decrease is intentionally not treated as the required 64-sample learning result. The 64-sample capability gate separately requires at least 1% relative mean-loss improvement under its fixed run.

Raw result: `/9950backfile/chenjiahui/evo_artifacts/results/qwen_process_seg/20261001_qps_v1/baseline_smoke_v2/result.json`.

## Pending gate

The process module remains blocked until the preregistered 64-sample QwenSeg-SAM31 capability run passes. No process SFT, pilot, or RL has started.

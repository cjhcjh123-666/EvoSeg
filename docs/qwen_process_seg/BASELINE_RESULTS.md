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

The first 64-sample run used the late Multiplex object-token residual. It failed the fixed 1% gate: four epochs/256 updates changed mean loss from 1.093123 to 1.091063 (−0.188%). A stronger one-sample/100-update audit also failed its 10% overfit threshold (1.058177 to 1.045150, −1.23%). Both negative results are retained under `baseline_capability64/` and `baseline_interface_one_sample/`.

That failure localized the issue to the late residual prompt interface. The corrected official grounding-token path (`Sam3Image.visual_prompt_embed`) passed a two-update real-mask smoke with a 13.12% decrease. The 64-sample capability gate must now be rerun on this corrected interface. No process SFT, pilot, or RL has started.

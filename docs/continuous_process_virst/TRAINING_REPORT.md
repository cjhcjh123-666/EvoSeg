# CPG-VIRST training report

## Executed stage

Only the preregistered capability stage was executed. The fixed pool contains
32 GroundMoRe official Sequential training expressions and 32 Long-RVOS
explicit-order training expressions. An epoch is the exact deterministic
interleave of those 64 expressions; no validation or test item is present.

- Seed: 11
- Frames per expression: 8, sampled by the official VIRST within-bin protocol
- Warm-up: 64 updates with `L_obj + 0.5 L_order + 0.01 L_len_ent`
- Joint SFT: 512 updates with the same losses plus official VIRST segmentation loss
- Learning rate: `1e-5`; trainable weights use FP32 masters
- Completed: 576/576 updates, status `success`
- Trainable parameters: 3,291,917
- Final process residual scale beta: 0.5013 (initialized near 0.5)
- Wall time: 1,805.85 seconds
- Peak allocated GPU memory: 19.587 GiB

The VIRST main VLM, large visual encoder, and official SAM2 pixel executor stayed
frozen. Only the compiler, process/video projections, segmental alignment,
pre-SegPrompter adapter, and fusion normalization were optimized. The official
VIRST checkout remained clean at `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`.

## Capability outcome

Paired by expression, segmentation loss changed by -0.4296 from the first to
last joint-SFT occurrence; 38/64 expressions decreased. The fixed-permutation
preference passed on 61/61 resolvable samples, and target object score exceeded
the best official distractor on 29/30 samples that had usable multi-instance
masks.

The GroundMoRe metadata audit found that `action_start/action_end` denotes the
whole queried-process/mask-validity interval rather than an unambiguous answer
clause. Clause-local interval supervision was therefore disabled instead of
creating a pseudo-label. This makes the complete preregistered Overfit Gate
fail and prohibits pilot launch.

Artifacts are under
`/9950backfile/chenjiahui/evo_artifacts/results/continuous_process_virst/20261001_cpg_sft/overfit_final_seed11/`.

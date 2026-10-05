# EvoSeg: active restart, 2026-10-06

This is the single current execution plan. Historical FTG/faithfulness specs
are retained for provenance, not instructions to restart their jobs.

## User-approved scope

Continue referring video object segmentation. Start from a strong, released,
fully trained segmentation VLM and keep its original SAM and inference path.
Do not replace it with SAM3.1 in the first run. Do not train Qwen from scratch,
resume the previous primary FTG job, or add the old factorization/verifier.

First candidate: `QuanzhuNiu/SaSaSa2VA-26B`, based on InternVL2.5-26B with the
released SAM2 grounding weights. Its public report gives 70.95 MeViS valid_u
J&F for the 26B Uniform mode. That is a **source-reported, legacy-protocol**
reference, not our MeViS-v2 result. The challenge's 67.45 is an ensemble score,
not this checkpoint's single-model baseline.

Sources:
- https://huggingface.co/QuanzhuNiu/SaSaSa2VA-26B
- https://arxiv.org/html/2509.16972v1 (Tables 2 and 3)

## One execution chain

1. Download the pinned public checkpoint through the user-authorized local
   `http://127.0.0.1:17890` proxy. Keep TLS verification. Do not send tokens.
2. Check full loading, including original SAM2, image normalization, prompt
   template, frame compression, SEG count and native video inference. No
   silently missing weights or reduced-resolution replacement of this path.
3. Establish the complete official baseline with fixed dataset version, split,
   aggregation, frame coverage, and absent-target handling. Single-model Uniform
   is the initial inference protocol; no ensemble or oracle prompts.
4. Fine-tune the released checkpoint using public train data only, retaining
   the native architecture. Use a small learning rate and an explicit held-out
   train-video check before spending the full multi-GPU budget. The original
   config loads *Sa2VA* as a precursor: do not launch it unchanged and pretend
   to be fine-tuning the released *SaSaSa2VA* checkpoint.
5. Keep source-reported, reproduced, and fine-tuned results separate. Improvement
   over our own baseline is not automatically SOTA or a method contribution.
   Add controlled experiments once the main result is strong and stable.

Immediate data candidate: MeViS-v2 train (already local). Image referring data
can preserve image grounding if its public train mixture is actually verified.
Do not silently introduce ReVOS training or claim zero-shot results on data
inherited from the released checkpoint. Ref-YT-VOS, Ref-DAVIS17, Long-RVOS and
ReVOS require verified evaluation protocols before joining the paper table.

## Locations and state

- Clean main worktree: `/tmp/EvoSeg-ftg-main`. The user's dirty original worktree
  at `/9950backfile/chenjiahui/EvoSeg` is untouched.
- New weights: `/9950backfile/chenjiahui/evo_artifacts/models/SaSaSa2VA-26B-official`.
- One active run: `/9950backfile/chenjiahui/evo_artifacts/results/evoseg_restart_20261006`.
- Asset preparation: `python -m projects.evoseg.restart.prepare_foundation --detach`.
  `ASSETS_READY` means download checks passed, not training or benchmark success.

## Native evaluation preparation

The released config and the repository's native model class construct successfully
under empty-weight initialization: 25,778,005,682 parameters and 1,838 state
tensors. This checks imports and architecture only, **not checkpoint loading**.
The download is approximately 92.04 GB, including mixed source tensor dtypes.

Use the existing `envs/virst/bin/python` with the independent
`envs/sasasa2va_native_overlay` directory first on `PYTHONPATH` (and the clean
repository root included). The overlay pins the model author's recommended
`transformers==4.42.3` and supplies missing native evaluation dependencies;
it does not modify the existing VIRST environment.

The official evaluation entry now accepts `--data-root`, defaults to the native
`uniform` mode, and reads selected-frame JSON only in `q_frame` mode. The initial
split root is `/9950backfile/chenjiahui/evo_artifacts/datasets/mevis_v2/valid_u`.
Pass `--expression-file` pointing to that split's `meta_expressions_v2.json`
explicitly; the local v2 metadata does not use the upstream v1 filename.
Ten asset/protocol unit tests pass. Native image transforms, frame compression,
SEG prompts and mask inference are unchanged. Do not launch full evaluation
until all shards are ready and complete loading has passed.

## Recoverable cleanup completed

Thirteen explicitly selected retired prototypes/invalid pre-repair exports were
renamed out of active `results/ftg` and `models` into
`/9950backfile/chenjiahui/evo_artifacts/archive/20261006_restart_obsolete`.
Its `MANIFEST.json` records each original path and restore destination.
No permanent deletion was performed and no disk space was reclaimed.

Protected: all datasets, released model weights, repaired exports, complete
61.95 baseline and 63.06 result, their source checkpoint, latest primary
checkpoint 101, historical evidence and uncommitted user files.

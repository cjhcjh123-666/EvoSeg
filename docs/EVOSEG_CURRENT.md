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

## User-requested overnight first run (09:00 Beijing target)

`python -m projects.evoseg.restart.overnight --detach` creates one detached,
locked supervisor. Inspect `OVERNIGHT_STATUS.json`, actual child PIDs and logs;
do not infer success from a launch marker. It never terminates foreign GPU jobs.
The first target is the nearest 09:00 Asia/Shanghai after launch (2026-10-06 for
this late-night session), reserving the last two hours for evaluation. Completion
by that target is an aim, not a measured runtime guarantee.

Chain: pinned assets ready -> strict complete native loading -> held-out native
diagnostic -> full MeViS-v2 907-expression baseline -> 20-update pilot -> held-out
gate -> continue the same optimizer/checkpoint up to 200 total updates or the
reserved evaluation window -> held-out and full 907-expression final evaluation.
A >3-point target-present held-out drop after the pilot withholds full training.
No score is called SOTA. An incomplete or failed stage records a failure instead
of silently advancing.

Fine-tuning is deliberately conservative: LoRA rank 32, alpha 64, dropout .05,
LR 2e-6, batch one video per GPU, accumulation two, native 100-frame training
wrap sampling and 20 compressed 448px images, ten 1024px keyframes, five public
expressions per video. Language LoRA, the released text-to-SAM projector and
native SAM mask decoder train; the original visual encoders, multimodal projector
and other SAM weights remain frozen. Native CE x2, Dice x.5 and assistant-token
language loss are retained. This is a **plain released-model fine-tune**, not the
discarded FTG method or a new paper contribution by itself.

The deterministic split contains 1579 fine-tune videos and 83 held-out videos.
Those held-out videos may have been seen by the released model's prior training.
Training sample and special-token alignment checks passed on actual public data.
Non-reentrant activation checkpointing is selected in the dedicated process to
support DDP unused-parameter handling without changing the native architecture.
The upstream evaluator omitted `<image>` for question expressions; MeViS-v2
has such questions. The new runner retains their exact public text and adds the
mandatory image placeholder identically for baseline and adapted inference.

Artifacts: `native_finetune/checkpoint_NNNNN.pth`, `LATEST.json`, training status,
per-expression full-frame prediction RLEs, before/after metrics, and
`FIRST_RUN_REPORT.json` only after all planned evaluation coverage passes.
Unit guards cover data splits, frame compression, loss gradients, deadline
calculation, unique evaluation shards and rejection of missing frame coverage.

## Recoverable cleanup completed

Thirteen explicitly selected retired prototypes/invalid pre-repair exports were
renamed out of active `results/ftg` and `models` into
`/9950backfile/chenjiahui/evo_artifacts/archive/20261006_restart_obsolete`.
Its `MANIFEST.json` records each original path and restore destination.
No permanent deletion was performed and no disk space was reclaimed.

Protected: all datasets, released model weights, repaired exports, complete
61.95 baseline and 63.06 result, their source checkpoint, latest primary
checkpoint 101, historical evidence and uncommitted user files.

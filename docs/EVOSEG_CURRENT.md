# EvoSeg: current multi-turn interaction project, 2026-10-06

## Active user-approved direction (supersedes the RVOS restart below)

The user explicitly approved moving to **multi-turn language-interactive
segmentation**: first pursue comparable public-benchmark improvements, while
preparing a small new dialogue-data pilot in parallel. Do not resume FTG,
faithfulness/refusal training or the completed RVOS repair runs.

Working method: **Scope-Controlled Conversational Segmentation (SCCS)**.
This is a research hypothesis, not an established novelty or SOTA claim.
The central distinction is between a historical mask used as a **witness**
("segment the person next to instance 1") and a historical mask authorized as
an **edit target** ("remove instance 1"). A model should infer the edit
operation, bind the correct history reference and predict the spatial change
scope, while preserving unaffected selections. Merely storing history or
adding mask tokens is not a new contribution: SegLLM and SAMTok already do so.

First foundation: released `zhouyik/Qwen3-VL-8B-SAMTok` with its original
SAMTok/SAM2.1 mask tokenizer and decoder. Keep the pixel foundation fixed in
the first stage. No scratch VLM or SAM3.1 swap. Public inference exports were
made lazy so native mask decoding does not require optional training frameworks;
no mask decoder weights or mathematical operations were changed.

The initial trainable prototype is `dialog/scope_head.py`: operation logits,
separate witness/edit history pointers, and instruction-conditioned spatial
scope. `dialog/state.py` provides auditable state/undo and scoped composition
primitives. These are tested scaffolds, not a trained end-to-end model. Training
integration, meaningful controls and novelty assessment remain required.

### Public experiments and fairness

- Main: official MR-RefCOCO, MR-RefCOCO+, MR-RefCOCOg; add MR-PACO when the
  exact image and annotation assets are complete.
- Published baselines: SegLLM and SAMTok. ConverSeg/ReasonSeg can supply
  single-turn reasoning checks, not substitute for multi-turn evaluation.
- PRIST is optional until complete data access and its exact protocol are verified.
- Some official scripts encode GT masks in conversational history. Reproduce
  this setting for like-for-like reported-table comparisons, and separately run
  prediction-history closed-loop evaluation. Never compare these as one score.
- Verify splits, per-round aggregation, full coverage, invalid-output handling,
  image/annotation-ID consistency, and inherited checkpoint training exposure.
- First essential controls: public native checkpoint; same-data ordinary LoRA;
  typed history binding without spatial scope; full SCCS. Larger training and
  SOTA claims require an actual measured advantage, not added module count.

Sources:
- https://github.com/berkeley-hipie/segllm/blob/main/DATASET.md
- https://github.com/berkeley-hipie/segllm/blob/main/scripts/eval/eval_mr_refcoco.sh
- https://github.com/bytedance/Sa2VA/tree/main/projects/samtok
- https://huggingface.co/zhouyik/Qwen3-VL-8B-SAMTok

### Parallel self-built data pilot

`dialog/build_edit_pilot.py` has generated **500 draft video dialogues / 2500
turns** from public MeViS-v2 train videos only, excluding the 83 prior fine-tune
held-out videos. Operations are select/add/remove/undo/replace, expressed by
deterministic templates and verbatim public referring expressions. Correct
masks are exact object-set unions of original annotation tracks, not model
predictions or generated masks. Object-level removal preserves retained targets
even where their pixel masks overlap.

Every record marks private supervision (not an inference input), source video,
original expression IDs, mask recipes and generated-language provenance.
`REVIEW_QUEUE.json` reserves 100 dialogues for review. **No human review has
been completed.** Drafts are neither an official benchmark nor human dialogues,
and are not yet mixed into primary public-benchmark training. Video use is a
later extension after the image multi-turn route is verified; do not silently
turn motion-dependent video descriptions into single-image instructions.

### Current assets and execution

- Run: `/9950backfile/chenjiahui/evo_artifacts/results/evoseg_dialog_20261006`.
- Foundation: `models/Qwen3-VL-8B-SAMTok-official`, pinned revision
  `b78aef1105d6a94049a2ba109f814dfcf21bec8e` (about 19.35 GB including pixel weights).
- Public conversations: `datasets/SegLLM-official`, pinned revision
  `848a4eb469ef8ea56e6d02f0e2c0f3f91eea61a5`.
- Draft data: `datasets/EvoSeg-Dialog-Pilot-v0` with `MANIFEST.json` and review queue.
- Preparation: `python -m projects.evoseg.dialog.prepare_assets --detach`.
  `ASSETS_DOWNLOADED` is only file inventory completion, not full model loading,
  benchmark reproduction or training completion.
- Initial GRefCOCO subset had incomplete RefCOCOg annotation coverage. Original
  COCO2014 instance GT was subsequently downloaded from the official S3 bucket
  over certificate-verified HTTPS. `PUBLIC_DATA_AUDIT_COCO_FULL.json` verifies
  all RefCOCO/+ /g train and validation targets and their image IDs. PACO assets
  still require their own source-image and part-annotation preparation. Never
  silently skip missing targets to improve reported metrics.
- Public validation inventory: MR-RefCOCO 1500 dialogues / 6832 rounds;
  MR-RefCOCO+ 1500 / 6765; MR-RefCOCOg 1263 / 3752. These are complete downloaded
  public files, not necessarily any sampled subset used in a different paper.
- Native first-round GPU smoke passed after strict language and specialized
  pixel-tokenizer loading. `NATIVE_SMOKE.json` is not a multi-turn baseline score.
  Initial missing mask-encoder keys while loading the *base* SAM checkpoint are
  expected; the complete specialized tokenizer state is then loaded strictly.
- Ten dialog unit tests pass, covering scope/reference gradients, immutable
  edits/undo, overlapping masks and public conversation parsing. The public
  `*_hard_train` files provide an external visual cue at turn one; those are
  valid visual-reference tasks, not unobserved conversational history.
- Before the matched pilot, no main method-training experiment had started. A separate **one-update**
  public-data gradient probe passed for native LoRA, history-role routing and
  spatial scope. `training_probe/PROBE_REPORT.json` is not a trained method score
  or a benchmark checkpoint; the edit branch has no supervised examples in it.

### Complete baseline and bounded matched pilot

Both full native history evaluations completed on 2026-10-06, each covering
17,349 rounds. Mean equal-dataset cIoU over rounds 2--6 is 81.8282% with GT
history and 78.0154% with generated history. These are released-model baselines,
not method gains or exact source-table/SOTA reproduction.

`python -m projects.evoseg.dialog.run_public_pilots --detach` runs two matched
64-update public-only pilots concurrently (4 GPUs each). Rank-8 LoRA, LR 1e-6,
accumulation 2, effective batch 8, seed 42, identical sample traces. Pixel weights
remain frozen. Images are held out by a stable filename hash across all three
datasets; all public validation images were already excluded from the cache.
Training samples one current assistant answer with GT history, up to round 6.
Controls: plain LoRA versus auxiliary witness pointer / operation / spatial
target alignment. **The auxiliary head is not used at inference and this is not
full SCCS.** Its inference effect is solely through trained language adapters.
No edit examples or generated drafts enter this pilot. This tests a component,
not the final novelty or the edit/preservation hypothesis.

After training, the same image-disjoint TRAIN holdout is evaluated closed-loop
for release / plain LoRA / auxiliary LoRA. Raw public markers are reconstructed
at evaluation, never GT-compiled training queries. Adapter hashes isolate
prediction caches and sample traces must match. A driver report is written only
after complete identical coverage; it does not automatically launch larger
training from loss alone. Results under `public_pilots_v1` are diagnostics, not
public validation benchmark scores. The released foundation's pretraining
exposure to these public training images remains possible.

The first matched pilots completed 64 updates each, with identical per-rank
sample traces. Both closed-loop diagnostics cover 69 dialogues / 237 rounds,
including 168 rounds after turn one:

| Variant | Multi-round gIoU | Pooled multi-round cIoU | Mean dataset/round 2--6 cIoU | Invalid / 237 |
|---|---:|---:|---:|---:|
| Released | 82.0325 | 84.2679 | 84.6353 | 4 |
| Plain LoRA | 80.8418 | 83.0583 | 77.0693 | 5 |
| Witness/scope auxiliary | 81.8635 | 83.2852 | 76.6389 | 6 |

Percentages; **train holdout diagnostic only, not benchmark scores**. The
round-6 equal-dataset average has only 7 examples total (2 / 4 / 1), making it
especially fragile. Auxiliary training is closer to release on pooled/gIoU but
is not consistently better across the predefined round metrics. No SOTA or
stable gain is established and larger training was not launched from these
results. Both pilot adapter checkpoints exist; pixel weights were not changed.

An evaluation-only follow-up runs the SAME checkpoints with GT history:
`run_public_pilots --detach --evaluation-only --history-mode gt_history
--checkpoint-root <public_pilots_v1> --output <public_pilots_v1_gt_history>`.
It makes no new optimizer updates. Compare both history protocols before
attributing the decline to lost segmentation capacity or history exposure.
The native baseline diagnostic visualization also distinguishes selected
previous-reference failures (sofa/laptop IoU 0) from a selected bus case where
the referenced previous mask was good (IoU .9564) but the current target failed.
This is evidence of different failure paths, not a causal proof of the method.

### Current multi-round baseline and training preparation

`dialog/eval_multiturn.py` evaluates all public raw validation dialogues with
the released native model, quantizer and mask decoder. Primary reporting is
foreground cIoU **by round**, with equal-dataset averages for MR-RefCOCO/+/g;
gIoU and invalid-output counts are included. Invalid output tokens contribute
an empty prediction and remain in the denominator, never silently disappearing.

History protocols are explicit and separate:

- `gt_history`: GT prior masks and GT prior answers are encoded with the native
  VQ mask-plus-normalized-bbox encoder.
- `predicted_history`: previous generated mask codes and actual assistant
  outputs are recycled. Unit tests forbid invoking GT-history callbacks in
  this mode; full predictions also record zero GT mask encodings for history.

Raw SegLLM mask/box markers are reconstructed as native VQ mask tokens. The
normalized box participates in mask encoding, not an invented language-level
coordinate format. Original SAMTok sampled/preprocessed files are not available
in the checked public data release, so exact source-table comparability remains
**pending**. Results here are reproducible full-public-file baselines, not an
automatic claim to reproduce Table 2 or SOTA. Main controls must use this exact
same protocol; externally reported numbers retain their source/protocol labels.

`python -m projects.evoseg.dialog.run_baselines --detach` completed its durable
chain: two complete-dialogue diagnostics -> eight-GPU GT-history evaluation of
all 17,349 rounds -> eight-GPU predicted-history evaluation of all 17,349 rounds.
Inspect `public_baselines/BASELINE_STATUS.json` and real child PIDs. The 16-round
diagnostic passed without invalid mask output, but its scores are not benchmarks.

`dialog/visualize_history.py` renders three largest history-sensitive failures
and three strongest closed-loop examples from saved predictions. The diagnostic
sheet is explicitly selected, not representative prevalence or method gain.
Inspected failures include switching to the opposite sofa/laptop and an
irrelevant bus; they demonstrate referent errors, not merely rough boundaries.
These examples alone do not distinguish earlier visual error propagation from
incorrect current-round history binding and cannot establish SCCS's benefit.

`dialog/cache_training.py` has prepared a bounded public-only pilot cache of
768 dialogues / 3024 rounds (256 dialogues from each referring dataset), excluding
all public validation images. It caches original-GT mask codes and frozen image
features, never synthetic masks or our unreviewed draft dialogues. Current public
supervision covers new referents and witness references; it does **not** validate
the edit/preservation branch of SCCS.

### Additional image edit drafts

The initial 500 video drafts remain unchanged. `dialog/build_image_edit_pilot.py`
has additionally generated 500 **image** draft dialogues / 2500 rounds using
public MRSeg training requests and original COCO masks. All 2701 distinct image
names present in public validation were excluded before choosing source pairs.
Instructions are templates plus verbatim original first-turn requests. Their
new/add/remove/undo/replace masks are exact object-set unions, not pixel subtraction
that would destroy an overlapping retained object. Partial-refine supervision is
not yet available. `IMAGE_REVIEW_QUEUE.json` and `IMAGE_EDIT_MANIFEST.json` both
mark human review pending. These drafts are not mixed into the main training run.

---

## Historical RVOS restart and negative-result provenance

The following plan and artifacts document the completed RVOS restart; they
are retained for provenance, not instructions to restart its jobs.

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
- Current repair run: `/9950backfile/chenjiahui/evo_artifacts/results/evoseg_restart_20261006_repair`.
- Completed first run / shared asset manifest:
  `/9950backfile/chenjiahui/evo_artifacts/results/evoseg_restart_20261006`.
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

## First-run finding and current repair

The first 200-update run completed all 907 validation expressions before 09:00.
Released native baseline: **67.6673 J&F**; adapted: **64.3350**, or -3.3323 points.
Target-present J&F fell 70.0882 -> 65.8563, while no-target rose 12.3045 -> 29.5436.
Across 51,123 ground-truth-present validation frames, empty predictions rose
from 1,963 (3.84%) to 3,917 (7.66%). This is a negative result, not SOTA.

Audit identified a real implementation mismatch: the custom fine-tune called
the HF predictor's SAM head, which hard-suppresses masks using object presence.
The author's training extension comments out that suppression so mask gradients
remain available for false-negative presence predictions. The first two sampled
training epochs also contained approximately 23% no-target expressions. Neither
finding alone establishes the entire causal attribution without a controlled run.

Repair execution: `python -m projects.evoseg.restart.repair_pilot --detach`.
The inference function and released weights are untouched. Training directly
calls the released prompt encoder and mask decoder, selecting masks by native
IoU but without inference-only suppression. Unit tests compare its tensors to
the exact author's training method. A real released-SAM GPU audit verifies
numerical parity, nonzero prompt gradients under forced negative presence, and
unchanged inference suppression; its dummy tensors are never used in training.

Two 50-update pilots start independently from the public release, keeping seed,
optimizer, frame compression, loss, batch and the 200-update LR schedule fixed:

- `ungated_only`: training-head fix with the original public expression sampler.
- `ungated_balanced10`: same fix with a 10% empty-expression sampling probability.

All selected expressions and masks remain public annotations. No labels are
fabricated; no synthetic data, refusal loss or extra presence head is introduced.
The expanded diagnostic spans all 83 held-out train videos with up to two positive
and one no-target expression per video. It is not a benchmark or unseen-pretraining
claim. Advancement requires overall and target-present diagnostic J&F drops <=.5
points and empty-on-present growth <=1 point. The selected qualified pilot is
checked again at 100 and 200 updates; failure stops continuation. Only a surviving
200-update run proceeds to the full 907-expression benchmark with the unchanged
native Uniform inference protocol and the verified released-baseline reference.

Old hard-gated checkpoints remain evaluable as historical evidence, but are
rejected as training resumes for the repaired semantics. The bounded repair
driver records real child PIDs, sampling fractions, checkpoints and guard results.

## Recoverable cleanup completed

Thirteen explicitly selected retired prototypes/invalid pre-repair exports were
renamed out of active `results/ftg` and `models` into
`/9950backfile/chenjiahui/evo_artifacts/archive/20261006_restart_obsolete`.
Its `MANIFEST.json` records each original path and restore destination.
No permanent deletion was performed and no disk space was reclaimed.

Protected: all datasets, released model weights, repaired exports, complete
61.95 baseline and 63.06 result, their source checkpoint, latest primary
checkpoint 101, historical evidence and uncommitted user files.

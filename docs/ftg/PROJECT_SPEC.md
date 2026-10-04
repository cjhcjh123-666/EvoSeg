# FTG: locked project specification

## Research question

FTG studies **Referring Video Object Segmentation with Segmentation VLMs**. The
project asks why models that already observe a video and emit frame-dependent
representations still degrade on motion-heavy, long-term, and dynamic referring
expressions.

The fixed hypothesis is that a monolithic spatiotemporal grounding state must
simultaneously preserve the referred object's identity and react to changing
object state. Too much invariance makes the prompt motion-insensitive; too much
adaptation permits identity drift. Temporal evidence can therefore be present in
the VLM without being exposed to the mask decoder through the right interface.

## Method claim

**Factorized Temporal Grounding (FTG)** separates one video-level persistent
identity from frame-dependent state observations. The released checkpoint's
pretrained segmentation prompt is retained as the persistent identity anchor;
an identity-conditioned state branch can only add a gated residual:

\[
z^{id}=F_{\mathrm{pretrained}}(q,V),\qquad
z_t^{state}=F_{state}(z^{id},H_t^{SAM}),
\]

\[
p_t=z^{id}+\sigma(G(z^{id},z_t^{state}))\odot R(z^{id},z_t^{state}),
\qquad M_t=\operatorname{SAM3}(I_{1:T},p_t).
\]

Here \(F_{state}\) is one identity-query cross-attention block over the native
SAM3 spatial feature map. \(R\)'s final projection is initialized to exactly
zero, so FTG is bit-for-bit the public foundation at initialization. Qwen, the
pretrained `[SEG]` projection, and SAM3 are frozen in the controlled pilot; only
the small state residual and gate are learned. The persistent prompt supplies a
stable identity base while the residual expresses current appearance and
location. The implementation adds no verifier, refusal path, candidate bank,
process compiler, GRU, RL objective, or post-hoc matcher.

The earlier randomly initialized two-token and broadly adapted implementations
are retained only as rejected controls. They changed the pretrained language-to-
instance geometry and failed on full MeViS-v2. Neither result is used as the
paper method.

## Fixed model and data

- VLM: Qwen3-VL-4B.
- Primary initialization: the public trained
  `Sa2VA-Qwen3-VL-4B-SAM3` checkpoint; no Faithful, refusal, verifier, or
  synthetic-data checkpoint is used.
- Pixel decoder: the checkpoint's native SAM3 video model, frozen for the first
  controlled study.
- Scratch initialization from `Qwen3-VL-4B-Instruct` plus official SAM3.1 is a
  controlled foundation ablation, not the main full-scale model.
- Default temporal budget: 16 uniformly sampled full-range frames.
- Pilot data: public Long-RVOS train plus public MeViS-v2 train.
- Full training mix: RefCOCO/+/g, Ref-Youtube-VOS train, MeViS-v2 train, and
  Long-RVOS train.
- No synthetic expressions, pseudo-labels, proprietary annotations, or generated
  training data.

## Controlled comparison

The primary strong-foundation comparison keeps the checkpoint, public data,
frame budget, frozen Qwen/SAM3 scope, loss, optimizer, steps, and evaluation
code fixed. Only the residual's access to persistent identity changes:

| Variant | Frozen identity base | State residual | Identity-conditioned state |
|---|---:|---:|---:|
| Public Foundation / Identity Only | yes | no | no |
| Unconditioned Frame Residual | yes | yes | no |
| ID + State without gate | yes | yes | yes |
| Anchored FTG | yes | yes | yes |

The decisive factorization comparison is **Unconditioned Frame Residual versus
Anchored FTG**; the public foundation establishes whether either learned
residual improves rather than merely changes predictions. The ungated variant
tests whether adaptive residual control is necessary. The broadly adapted
Frame Prompt and two-token FTG runs are retained as explicitly labeled negative
diagnostics, not parameter-matched primary controls. The scratch-Qwen suite
still reports Global/Frame/Identity/State/FTG under an exactly shared trainable
scope as a separate foundation ablation.

## Evaluation

Main benchmarks are Ref-Youtube-VOS, MeViS-v2, Long-RVOS, and Ref-DAVIS17.
Long-RVOS must report Static, Dynamic, Hybrid, and Overall results separately,
including J&F, tIoU, and vIoU. ReVOS is a reasoning-generalization table;
GroundMoRe is supplementary and neither dataset is used for primary method
selection.

External comparisons should cover segmentation VLMs (Sa2VA, VideoLISA, VISA,
VRS-HQ, InstructSeg, VIRST, StreamingRVOS), specialist RVOS models (ReferDINO,
SAMWISE, DeRVOS), and the Long-RVOS baseline ReferMo. Claims must distinguish
reported literature numbers from results reproduced in this repository.

## Claim boundary

The intended evidence is selective: gains should be larger for Dynamic/Hybrid
and MeViS motion expressions than for Static expressions, while Static should
not regress materially. A uniform gain alone does not establish the identity-
state hypothesis. The temporal-budget ablation is T=8/16/32; the order diagnostic
is Original/Shuffle/Reverse.

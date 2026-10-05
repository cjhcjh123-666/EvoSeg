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
identity from frame-dependent state observations. The released VIRST
SegPrompter is first run unchanged, then its prompts are factorized into a
temporal mean and a zero-mean residual:

\[
z^{id}=\frac{1}{T}\sum_t x_t,\qquad z_t^{state}=x_t-z^{id},
\]

\[
p_t=x_t+\operatorname{center}_t\left(
\rho\tanh G([z^{id};z_t^{state}])\odot z_t^{state}\right),
\qquad M_t=\operatorname{SAM2.1}(I_{1:T},p_t).
\]

The gate's final projection is zero-initialized, so step zero is bit-exact to
public VIRST. Re-centering guarantees that the temporal mean remains the
persistent identity throughout training. The complete public
VideoChat/VIRST/SAM2.1 foundation and keyframe scores are frozen; only the small
composer is learned. The persistent mean supplies a stable identity base while
the centered residual expresses current appearance and location.
The implementation adds no verifier, refusal path, candidate bank, process
compiler, GRU, RL objective, or post-hoc matcher.

The earlier randomly initialized two-token and broadly adapted implementations
are retained only as rejected controls. They changed the pretrained language-to-
instance geometry and failed on full MeViS-v2. Neither result is used as the
paper method.

## Fixed model and data

- VLM foundation: public VIRST (VideoChat-Flash Qwen2-7B).
- Pixel decoder: VIRST's native SAM2.1 video model.
- Primary initialization: the released VIRST checkpoint; no Faithful, refusal,
  verifier, synthetic-data, or generated-data checkpoint is used.
- The complete public foundation is frozen for the first controlled stage; only
  the exact-initialized FTG composer is updated.
- Qwen3-VL-4B + SAM3.1 remains a foundation ablation and negative interface
  study, not the primary full-scale model.
- The full-run temporal-budget ablation is T=8/16/32.
- Pilot data: public Long-RVOS train plus public MeViS-v2 train.
- Full training mix: RefCOCO/+/g, Ref-Youtube-VOS train, MeViS-v2 train, and
  Long-RVOS train.
- No synthetic expressions, pseudo-labels, proprietary annotations, or generated
  training data.

## Controlled comparison

The primary strong-foundation comparison keeps the VIRST checkpoint, public
data, frame budget, frozen foundation, loss, optimizer, steps, and evaluation
code fixed. Only prompt factorization/composition changes:

| Variant | Frozen identity base | State residual | Identity-conditioned state |
|---|---:|---:|---:|
| Public VIRST | implicit | released frame prompt | n/a |
| Identity Only | yes | no | no |
| State Only | no | yes | no |
| Factorized without gate | yes | yes | no learned gate |
| Unconditioned FTG | yes | yes | no |
| Scalar FTG | yes | yes | yes, scalar |
| **Channel-gated FTG** | yes | yes | yes, vector |

The decisive final comparison is **public VIRST versus channel-gated FTG** from
an exactly identical step-zero function. Identity/State controls validate the
roles, while unconditioned and scalar gates isolate identity conditioning and
elementwise composition. The Qwen suite is reported separately as a foundation
ablation.

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

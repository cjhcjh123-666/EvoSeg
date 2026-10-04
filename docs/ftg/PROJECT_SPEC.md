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
identity from frame-dependent state observations. The two factors are no longer
summed into one detector-prompt vector. They enter SAM3 as distinct sparse
tokens:

\[
z^{id}=F_{id}(q,\operatorname{Pool}_t H_t),\qquad
z_t^{state}=F_{state}(z^{id},H_t^{SAM}),
\]

\[
p_t=[z^{id};z_t^{state}],\qquad
(M_t,m_t)=\operatorname{Track}(I_t,m_{t-1};p_t).
\]

Here \(F_{state}\) is one identity-query cross-attention block over the native
SAM3 spatial feature map. The first sparse token is identical at every frame;
only the second token changes. The persistent token supplies stable ownership,
the state token supplies evidence about current appearance and location, and
the native tracker carries both into recurrent memory. The implementation adds
no verifier, refusal path, candidate bank,
process compiler, GRU, RL objective, or post-hoc matcher.

The earlier additive prompt implementation is retained only as a rejected
control. On the scaled three-seed pilot it underperformed Frame Prompt in every
seed, and direct training-free insertion of a Qwen mask into SAM3.1 tracker
memory also failed. Neither result is used as the paper method.

## Fixed model and data

- VLM: Qwen3-VL-4B.
- Primary initialization: the public trained
  `Sa2VA-Qwen3-VL-4B-SAM3` checkpoint; no Faithful, refusal, verifier, or
  synthetic-data checkpoint is used.
- Pixel decoder and persistent memory: the checkpoint's native SAM3 video
  tracker, frozen for the first controlled study.
- Scratch initialization from `Qwen3-VL-4B-Instruct` plus official SAM3.1 is a
  controlled foundation ablation, not the main full-scale model.
- Default temporal budget: 16 uniformly sampled full-range frames.
- Pilot data: public Long-RVOS train plus public MeViS-v2 train.
- Full training mix: RefCOCO/+/g, Ref-Youtube-VOS train, MeViS-v2 train, and
  Long-RVOS train.
- No synthetic expressions, pseudo-labels, proprietary annotations, or generated
  training data.

## Controlled comparison

Every variant uses the same public pretrained checkpoint, data manifest,
frame budget, LoRA placement, loss, optimizer, steps, and evaluation code. Only
the grounding interface changes:

| Variant | Persistent identity | Dynamic state | Frame-specific prompt |
|---|---:|---:|---:|
| Pretrained Monolithic | mixed | mixed | no |
| Frame Prompt | mixed | mixed | yes |
| Identity Memory | yes | no | no |
| State Only | no | yes | yes |
| ID + State, vector sum | yes | yes | yes |
| FTG memory interface | yes | yes | yes |

The decisive comparison remains **Frame Prompt versus FTG** with identical
initial weights and supervision. It isolates persistent identity memory from
the generic benefit of frame-specific prompts. Identity Memory and State Only
test each factor alone; the rejected vector-sum variant tests whether gains
require distinct interfaces.

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

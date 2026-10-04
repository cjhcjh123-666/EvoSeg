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

**Factorized Temporal Grounding (FTG)** separates the grounding representation
into one video-level persistent identity token and frame-dependent state tokens.
Each state is conditioned on the fixed identity. Frozen SAM3.1 native text
features provide the pretrained semantic anchor, while FTG assigns the two
factors distinct jobs: the dynamic state supplies a gated frame-specific prompt
residual, and the persistent identity associates the referred object with SAM's
candidate object-query features:

\[
z^{id}=F_{id}(q,\operatorname{Pool}_t H_t),\qquad
z_t^{state}=F_{state}(H_t,z^{id}),
\]

\[
p_t^{dyn}=\sigma(W_g[z^{id};z_t^{state}])\odot
W_{state}z_t^{state},
\qquad
s_{t,k}=s^{SAM}_{t,k}+\tau\cos(Az^{id},h^{SAM}_{t,k}).
\]

Here the selected mask is \(M_{t,\arg\max_k s_{t,k}}\). Frame Prompt receives
the same association head but replaces the persistent identity with its
frame-dependent monolithic representation, keeping the decisive comparison
controlled. This is representation factorization, not another temporal
aggregation module.
The implementation adds no verifier, refusal path, candidate bank, process
compiler, GRU, RL objective, keyframe classifier, or post-hoc matcher.

## Fixed model and data

- Backbone: Qwen3-VL-4B-Instruct.
- Pixel decoder: frozen official SAM3.1 Object Multiplex model.
- Interface: official `visual_prompt_embed` grounding-token path.
- Default temporal budget: 16 uniformly sampled full-range frames.
- Pilot data: public Long-RVOS train plus public MeViS-v2 train.
- Full training mix: RefCOCO/+/g, Ref-Youtube-VOS train, MeViS-v2 train, and
  Long-RVOS train.
- No synthetic expressions, pseudo-labels, proprietary annotations, or generated
  training data.

## Controlled comparison

Every variant uses the same Qwen checkpoint, SAM3.1 checkpoint, data manifest,
frame budget, LoRA placement, loss, optimizer, steps, and evaluation code. Only
the grounding interface changes:

| Variant | Persistent identity | Dynamic state | Frame-specific prompt |
|---|---:|---:|---:|
| Global Prompt | mixed | mixed | no |
| Frame Prompt | mixed | mixed | yes |
| Identity Only | yes | no | no |
| State Only | no | yes | yes |
| ID + State, no gate | yes | yes | yes |
| FTG | yes | yes | yes |

The decisive comparison is **Frame Prompt versus FTG**. It isolates explicit
identity persistence from the generic benefit of frame-specific prompts.

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

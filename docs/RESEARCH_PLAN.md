# EvoSeg — Research Plan (locked 2026-08-13)

## Working title
**Faithful Referring Segmentation in Image & Video: Learning to Segment, Abstain, Verify, and Clarify**

## One-paragraph story
Existing VLM-driven referring segmentation systems hallucinate systematically on
empty-target / counterfactual / video-temporal-misaligned queries: they emit a mask
even when the query refers to nothing, or to the wrong object. We build a unified
lightweight (4B/8B) image+video referring segmenter that learns **when to segment,
when to abstain, when to verify, and when to ask for clarification**, and we
contribute the **first video referring hallucination benchmark**.

## Contributions (4 blocks)
1. **Method — Faithful RL + verification:**
   - Faithfulness-aware GRPO rewards: empty-target abstention reward, query-swap
     detection reward, mask IoU reward — unified over image and video.
   - A learned query-mask consistency **verifier** drives selective test-time
     compute: uncertain -> refine -> still wrong -> abstain.
   - **Clarification interaction**: when the query is ambiguous (multi-hit / low
     confidence / no-hit), the model asks one clarifying question before segmenting.
2. **Benchmark — first video referring hallucination benchmark:**
   - Built on MeViS + Ref-YT-VOS + Ryvos with three query types:
     empty-target / counterfactual / temporal-misalignment.
   - Metrics: abstention correctness (N-acc style), hallucination severity,
     disentangled vision- vs language-driven failures.
3. **Accuracy:** RefCOCO/+/g ~= Sa2VA-1B (79.6/73.6/77.7); video J&F ~= Sa2VA-i-1B;
   empty-target N-acc beats current SOTA.
4. **Lightweight (later):** 4B base + distillation/quantization, accuracy-efficiency curves.

## Bases (3)
- Qwen3-VL-4B-Instruct  (lightweight track)
- Qwen3-VL-8B-Instruct  (main track)
- VILA1.5-3B            (efficiency baseline / method generality check)

## Codebase strategy
- Fork of ByteDance **Pixel-LLM (Sa2VA)** — already supports Qwen3-VL + SAM2/SAM3
  training & eval. Do NOT port into EvoVILA (VILA-specific + AGENTS.md constraints).
- Our additions live under `projects/evoseg/`.

## Training recipe (SFT stage, aligned with Sa2VA-3B config)
- Data: RefCOCO(×5) + RefCOCO+(×5) + RefCOCOg(×4) + gRefCOCO + ReasonSeg
  + video (ReVOS/MeViS/Ref-YT-VOS/Ryvos) + **LLaVA-1.5-665K VQA (anti-forgetting)**
  + GCG (GranDf/Flickr30K/OpenPSG) — concat + uniform sampling, 1 epoch.
- Loss: mask BCE (sigmoid CE) w=2.0 + Dice w=0.5.
- LLM: freeze + LoRA r=128/alpha=256 (+ lm_head/embed_tokens); vision frozen.
- SAM2 decoder: trainable (frozen_sam2_decoder=False). Image 1024.
- lr 4e-5, wd 0.05, warmup 0.05, cosine; bs 2 x accum (eff. 128 on 8x A800... adjust).

## Faithfulness stage (GRPO, ~4-8 x A800 feasible; Dr. Seg ran 7B on 4x H800)
- Rewards: r_abstain (empty-target), r_swap (query-swap detection), r_iou, r_fmt.
- Verifier: query-mask consistency scoring head -> drives refine/abstain/clarify.
- Video consistency: train/inference identical (Sa2VA-i lesson); uniform frame sampling.

## Benchmark curation (video hallucination)
- SA-1B already extracted locally (100 shards) -> source for counterfactual masks.
- Query types: empty-target (query refers to absent object), counterfactual
  (swap object attr), temporal-misalignment (object before/after query state).
- Verification loop: VLM verify (mask <-> query) + human spot-check.

## Timeline (target CVPR 2027, deadline ~Nov 2026)
- W1-2   : repo setup + Qwen3-VL-4B/8B download + data pipeline
- W3-5   : SFT baseline -> RefCOCO/+/g >= 79/73/77
- W6-8   : faithful RL + verifier + clarification; gRefCOCO N-acc SOTA
- W6-9   : (parallel) video hallucination benchmark
- W9-11  : full experiments + baselines + ablations
- W11-13 : paper

## Baselines (paper main table)
LISA / GLaMM / PixelLM / GSVA / OMG-LLaVA / Sa2VA-1B / Sa2VA-i-1B /
CONVERSEG-NET-3B / PSALM-1.3B / (+ RL upper bound: LENS-3B, Dr. Seg)

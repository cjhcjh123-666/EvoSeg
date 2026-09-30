# VIRST baseline reproduction

Status: **PASS**. This is a reuse-and-recompute check of the completed official
VIRST run; no segmentation inference was repeated.

## Frozen implementation and weights

- Official checkout: `/9950backfile/chenjiahui/evo_artifacts/external/VIRST`
- Official commit: `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`
- Checkout state at audit: clean
- VIRST checkpoint: `checkpoints/virst_checkpoint.pt`
- VIRST checkpoint SHA-256: `bd4a70841c1334e77669265b0a0dbd3e580a46e2f0b739e2f48cf8e6cd3ba8a4`
- SAM2.1 Hiera-L checkpoint SHA-256: `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`
- VideoChat index SHA-256: `6ee2dfa85c4ef29f24093cfdd3b076eba89443cb4461e4b5ba237a026cccd574`

The official checkout and checkpoints were not modified.

## Long-RVOS paired validation

Source result:
`/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260922_virst_long_rvos_all274_sharded`

Protocol and integrity:

- 1,189 / 1,189 official expressions succeeded; failures: 0
- 90 source videos, 274 paired objects
- VLM input: 64 sampled frames; SAM prompt input: 32 frames; official full-video propagation
- same-video sampling mismatches: 0
- GT masks exposed to model input: 0
- metrics are expression-averaged within object and then object-averaged within description type

| Type | Expressions | Objects | J | F | J&F |
|---|---:|---:|---:|---:|---:|
| Static | 425 | 274 | 59.24 | 59.50 | 59.37 |
| Dynamic | 418 | 274 | 58.75 | 59.25 | 59.00 |
| Hybrid | 346 | 263 | 58.76 | 59.26 | 59.01 |

Dynamic minus Static J&F is **-0.37 percentage points**. The existing 2,000-resample
source-video cluster bootstrap interval is **[-2.68, +1.83] pp**. This exactly
matches the prior cross-model audit and passes the prerequisite for ProcessVIRST.

## GroundMoRe baseline status

The official VIRST repository has no GroundMoRe dataset adapter or evaluator.
GroundMoRe is therefore not claimed as reproduced in this checkpoint. A separate
adapter must preserve the official temporal interval and mask protocol before a
baseline number is valid.

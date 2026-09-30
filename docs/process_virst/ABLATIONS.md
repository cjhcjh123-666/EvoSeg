# ProcessVIRST SFT ablations

All ablations use identical frozen VIRST/SAM2 checkpoints, GroundMoRe pilot
expressions, frames, spatial resolution, and evaluator.

| Ablation | Monotonic | Order loss | J&F | Reverse margin | BlockSwap margin |
|---|---|---|---:|---:|---:|
| Official VIRST | no process module | no | 27.731 | N/A | N/A |
| Global process attention | no | yes | 27.828 | 0 | 0 |
| Monotonic process | yes | no | 27.851 | +0.00002 | -0.00055 |
| Full ProcessVIRST seed11 | yes | yes | 27.876 | +0.00235 | +0.00299 |

The matched-seed improvements of Full over no-order (+0.025 pp) and global
(+0.047 pp) both have bootstrap intervals crossing zero. Across all three Full
seeds the sign of order preference is inconsistent. Therefore neither the
monotonic structure nor `L_order` has demonstrated a reproducible segmentation
benefit in this SFT pilot.

The no-monotonic control's exactly zero alignment margin is an implementation
sanity check: its global score is permutation invariant, as also covered by a
synthetic unit test.

# ProcessVIRST SFT training record

## Frozen configuration

- Official VIRST checkout: `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`
- VIRST checkpoint SHA256: `bd4a70841c1334e77669265b0a0dbd3e580a46e2f0b739e2f48cf8e6cd3ba8a4`
- SAM2.1 Hiera-L checkpoint SHA256: `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`
- Main VLM, visual encoder, official SegPrompter, and SAM2 are frozen.
- Trainable: Process Compiler, event/frame alignment, process adapter, positive `beta`, and fusion LayerNorm.
- Trainable parameters: 3,217,666 (3,217,154 process parameters plus 512 fusion LayerNorm parameters).
- Optimizer: AdamW, learning rate `1e-5`, no weight decay, gradient norm cap `1.0`.
- Input: eight temporally ordered frames per training sample; four process slots.
- Loss: official VIRST segmentation loss; verified-order samples additionally use margin `0.2` with weight `1.0`.
- Fixed seeds: 11, 23, and 42. Final validation is never used to select a seed.

## Data audit

- Long-RVOS pilot training view: 64 official-train source videos, 626 expressions, 183 mask tracks.
- GroundMoRe Sequential trainval available locally: 413 videos and 1,171 expressions.
- The official metadata contains 414 Sequential videos and 1,173 expressions. Clip `LM-JY2eoFp8_1534_1546` (two expressions) is absent from the local official data copy and is recorded as missing rather than silently counted.
- Long-RVOS explicit-order examples are selected only by the preregistered deterministic expression filter.
- GroundMoRe masks are zero outside the official action interval and use the official 6 FPS interval conversion.

## Smoke checks

- Official VIRST + ProcessVIRST full forward/backward succeeded for six steps.
- Segmentation loss decreased from 0.814 to 0.565 in the warm-up smoke.
- Fixed-permutation order loss decreased from 0.202 to 0.194 over four ordered steps.
- `beta` remained approximately 1.0; peak allocated GPU memory was 19.38 GiB.
- A separate two-step loader smoke succeeded with both Long-RVOS and GroundMoRe registered simultaneously.

### Mixed-precision correction

The first 3-seed pilot launch was stopped at roughly 160/384 updates after an audit found that a blanket `model.to(bfloat16)` had also converted the new trainable head. With learning rate `1e-5`, `beta` remained bit-identical throughout those partial runs. Their logs are retained, but they are marked numerically invalid and are excluded from all results.

The implementation now keeps the frozen VIRST/SAM2 model in BF16 while retaining FP32 master parameters for the ProcessVIRST head and fusion LayerNorm. A nine-update real-data re-smoke confirmed:

- every trainable tensor is FP32;
- `beta` changes across updates;
- trainable-parameter delta L2 is 0.06135 and max absolute delta is 0.0000877;
- 3,217,620 of 3,217,666 trainable scalar values changed;
- peak allocated memory remains 19.39 GiB.

The preregistered pilot is restarted from scratch in new output directories; no partial BF16 checkpoint is resumed.

## Pilot completion and resumability

All three valid FP32 pilot seeds completed 384/384 updates successfully on
2026-09-30 at 20:56--20:57 Asia/Shanghai. Their final beta values are 1.00295,
1.00314, and 1.00313 for seeds 11, 23, and 42; peak allocated memory was about
19.39 GiB per worker. The ablation jobs, official GroundMoRe full baseline,
data extraction, and status monitor remained active after the main seeds
finished. An initial unprivileged status probe could not access the tmux socket
or NVIDIA device nodes; a privileged read-only check confirmed that the jobs
and driver were healthy. No infrastructure failure is inferred from that
sandbox access error.

The trainer now writes an atomic `resume_latest.pt` every 32 completed updates.
It includes the ProcessVIRST and fusion weights, optimizer state, initial
weights, next update, and Python/NumPy/Torch/CUDA RNG states. Resume rejects any
change to the seed, update budget, frame count, learning rate, alignment mode,
or order-loss condition. This protects future runs from losing the entire
preregistered trajectory during another external interruption.

## Preregistered pilot budget

Before viewing pilot results, the three-seed pilot was fixed to 128 identity-warm-up updates followed by 256 ordered-SFT updates per seed. Datasets are sampled 1:1. The learning rate, frame count, loss weights, slot count, and architecture are identical across seeds.

If the pilot gate is directionally satisfied, the full fixed budget is one 1,000-update warm-up epoch plus three 1,000-update ordered epochs. No hyperparameter sweep is permitted between pilot and full training.

## Current status

The two-video GroundMoRe protocol check completed for all six expressions. Official VIRST scored 29.31 J&F; the six-step smoke checkpoint scored 29.61 J&F. This +0.30 point difference is only an integration check and is not treated as scientific evidence.

## Completed preregistered pilot training

All three Full seeds and both seed-11 controls completed 128 warm-up plus 256
ordered-stage updates. No final-validation seed selection was performed.

| Condition | Seed | Updates | beta | Peak allocated GPU memory | Elapsed |
|---|---:|---:|---:|---:|---:|
| Full monotonic + order loss | 11 | 384 | 1.00295 | 19.393 GiB | 43.1 min |
| Full monotonic + order loss | 23 | 384 | 1.00314 | 19.392 GiB | 42.5 min |
| Full monotonic + order loss | 42 | 384 | 1.00313 | 19.392 GiB | 42.1 min |
| Monotonic, no order loss | 11 | 384 | 1.00293 | 19.393 GiB | 38.4 min |
| Global/non-monotonic | 11 | 384 | 1.00297 | 19.393 GiB | 38.2 min |

Trainable parameters: 3,217,666, of which 3,217,154 are in the Process
Conditioner and 512 are the fusion LayerNorm. Frozen VIRST and SAM2 parameters
were not updated. The FP32-master check found more than 3.217 million changed
trainable values for every Full seed.

The final verified-order training margins remained near zero and the order loss
remained near its initial 0.2 hinge value. This is reported as a negative
optimization outcome under the fixed pilot budget; no learning-rate, lambda,
margin, epoch, or seed search was performed.

## Verification

- `git diff --check`: pass.
- Python compilation of the trainer, pilot summarizer, and plotting code: pass.
- 18 ProcessVIRST tests invoked directly: pass, including the synthetic
  monotonic/order tests and resume-configuration tests.
- The VIRST environment does not contain the `pytest` package
  (`No module named pytest`), so the required pytest command was attempted but
  could not collect; no package was installed into the frozen environment.

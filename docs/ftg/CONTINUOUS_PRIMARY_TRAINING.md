# Continuous primary training (approximately 20 hours)

The user requests genuine continuous eight-GPU training, not GPU reservation
through idle memory holders, and asks to inspect quality before queuing work.

## Checkpoint-100 findings

The eight-expression monitor in the initial run reports J&F 45.33, but covers
only two distinct videos. It is not a public benchmark result or adequate
evidence of improvement. Surfboard boundaries look relatively good; the snowy
boy examples show referent switches and background false positives. The
motion-dependent surfer expression fails in this small check. No SOTA claim.

The initial run failed after development evaluation with DDP missing-gradient
errors. Checkpoint 100 was saved successfully. Missing gradients concentrate
on float32 Linear/LoRA weights while normalization weights retain gradients,
suggesting an autocast/no-grad cached-cast interaction; this requires runtime
verification, not merely enabling DDP unused-parameter detection.

The fix disables cached autocast in train/development, clears the cache after
evaluation, and performs development forward passes through DDP on every rank.
Every trainable parameter is checked for a gradient. A resumed run evaluates
immediately after its first update and must then successfully continue training.

## Persistent queue

Use the same initialized primary model and optimizer throughout, with no CUDA
teardown between these stages. Continue from saved checkpoint 100, do not start
an ablation or another seed.

| Phase | Endpoint | Work |
|---|---:|---|
| First epoch recovery | 2,876 updates | Complete the initial public MeViS-v2/Long-RVOS pass |
| Main convergence | 8,628 updates | Continue the main model through epochs 2–3 |
| Low-LR refinement | 14,380 updates | Continue through epochs 4–5 on the same cosine schedule |

The five-epoch horizon replaces the previous one-epoch horizon; this is an
expanded main training budget, not a controlled baseline comparison. Save
weights every 100 updates; evaluate 32 distinct held-out videos every 500
updates, balancing the two sources and cycling Long-RVOS expression types.
Inspect qualitative examples as well as scores. These remain sampled train
holdout metrics, never official benchmark scores.

An explicit 20-hour wall limit saves the current checkpoint before ending.
Actual completion can be earlier if five epochs finish sooner, or finish fewer
epochs if shared-GPU contention slows the run. QUEUE_PLAN.json and STATUS.json
record phases and progress; failure stops the queue rather than burning GPU
time with dummy workloads.

## Resource limitation

Persistent genuine training prevents this job from intentionally releasing its
CUDA allocations between stages. It does not guarantee GPU exclusivity on a
shared server. No Slurm/PBS executable was found in the current PATH; an admin
or a real cluster reservation is required to guarantee an exclusive interval.
Do not kill other users' processes or change GPU compute modes.

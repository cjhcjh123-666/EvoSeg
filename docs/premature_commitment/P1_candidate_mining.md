# M5.1 — Ambiguous-referent candidate pool

**Scope**: build a pool of *real* referential-ambiguity cases that a human
annotator would plausibly find hard, so that P2 (prefix-identifiability
annotation) has something meaningful to label.

**Hard constraint from Issue #3**: no toy negatives. The miner uses **only**
real Ref-YT-VOS `valid` expressions and the dataset's own instance GT tracks.
It does **not** create random absent classes, `dog → elephant` substitutions,
category-word replacements, or any other synthetic corruption.

**Artifacts**

| item | path |
|---|---|
| miner | `projects/evoseg/premature_commitment/mine_ambiguous_candidates.py` |
| full pool (250 cases) | `<artifact_root>/results/premature_commitment/candidate_pool.json` |
| flat summary | `<artifact_root>/results/premature_commitment/candidate_summary.csv` |
| dataset statistics | `<artifact_root>/results/premature_commitment/candidate_stats.json` |
| local contact sheet | `<artifact_root>/results/premature_commitment/candidates.html` |
| committed statistics copy | `docs/premature_commitment/results/candidate_stats.json` |

`<artifact_root>` = `/9950backfile/chenjiahui/evo_artifacts` (default; override
with `--out-dir`).

**Reproduce**

```bash
export PYTHONSAFEPATH=1
python projects/evoseg/premature_commitment/mine_ambiguous_candidates.py \
  --out-dir /9950backfile/chenjiahui/evo_artifacts/results/premature_commitment \
  --pool-size 250 --top-k 80
```

Runtime ≈ 7.5 min for the 53 videos with GT instance tracks that are reachable
from the Ref-YT-VOS `valid` annotations present locally. The pipeline has no
stochastic step and is deterministic (`--seed` is accepted for interface
uniformity and prints a notice that it is unused).

---

## 1. Mining signals (per expression)

All signals are computed on **annotated frames only** (`meta_expressions`
`frames` list), never on raw unannotated video frames (Ref-YT-VOS annotates
every 5th frame). Instance masks are recovered through the dataset's own
per-`obj_id` expression directories; note that Ref-YT-VOS keys annotation
directories by **expression id**, so one instance's masks are shared by all of
its expressions (handled in `utils.py` / `load_masks`).

| signal | meaning |
|---|---|
| `n_same_cat` | # other instances sharing the target's category |
| `n_other` | # other instances in the video (any category) |
| `co_visible_frames` | # frames in which the target **and** ≥1 same-category distractor are both present |
| `co_visible_ratio` | `co_visible_frames / n_target_frames` (per-frame, ≤ 1) |
| `min_centroid_dist_norm` | smallest normalised centroid distance between target and a same-category distractor over all co-visible frames |
| `max_dilated_overlap` | max IoU between the target and a distractor after dilating the target by ≈2% of the shorter side — a proxy for touching/crossing instances. Direct mask overlap is impossible here because Ref-YT-VOS instance masks are disjoint by construction. |
| `target_gap_frames` | # absent annotated frames strictly between the target's first and last present frame (disappearance → reappearance) |
| `target_absent_frames` | total absent annotated frames |
| `query_type` | dataset-derived query family from a fixed vocabulary (see below) |

`query_type` is a bag-of-words score over four hard-coded vocabularies; the
winning family is prefixed `multi_` when it fires on ≥2 tokens. Families:
`appearance`, `motion`, `relation`, `temporal_order`, `multi_<family>`,
`other` (no vocabulary hit). This is a *descriptive* label for stratification,
not a semantic parser.

## 2. Ranking

```
rank_score = 1.0*z(co_visible_ratio)
           + 0.8*z(-min_centroid_dist_norm)
           + 0.8*z(max_dilated_overlap)
           + 0.6*z(n_same_cat)
           + 0.5*z(target_gap_frames)
           + 0.4*z(1[query_type not in {other, appearance}])
```

`z(·)` is a population z-score over the mined cases, with non-finite values
treated as 0. **Baseline identity failure is deliberately *not* part of the
score.** It is carried as an optional, separate column and never used for
ranking, precisely to avoid the selection bias the issue warns about
(“baseline 失败只能作为其中一个 ranking signal，不能只挑 baseline 错误样本”).

The pool is the top `--pool-size` by score. The annotation subset (`--top-k`)
is then drawn with **soft quotas over `query_type`** (proportional to the
mined distribution, floor 3), with rank backfill, so the human pool is not
dominated by a single query family.

## 3. Pool statistics (250 cases)

| statistic | value |
|---|---|
| cases mined | 834 |
| pool size | 250 |
| top-k (annotation subset) | 80 |
| videos | 53 |
| median co-visible frames (pool) | 20.0 |
| median co-visible ratio (pool) | 1.00 |
| share of pool with ≥2 same-category distractors | 58.4% |
| median co-visible frames (top-80) | 23.5 |
| share of top-80 with ≥2 same-category distractors | 72.5% |
| cases with a disappearance gap | 72 |
| cases with ≥1 absent annotated frame | 186 |
| cases with non-zero dilated overlap | 252 (all of the top-80) |

`n_same_cat` distribution in the pool: `1 → 104`, `2 → 120`, `3 → 16`,
`4 → 10` (every pooled case has at least one same-category competitor).

`co_visible_ratio` across the mined cases: mean **0.406**, median 0.0, max
36 frames. `co_visible_frames`: mean 9.87.

Query-type distribution in the pool:

| family | n |
|---|---|
| `multi_relation` | 153 |
| `relation` | 39 |
| `multi_appearance` | 21 |
| `motion` | 18 |
| `temporal_order` | 8 |
| `other` | 6 |
| `multi_motion` | 3 |
| `appearance` | 2 |

Top-80 (the annotation subset): `multi_relation` 55, `motion` 11,
`multi_appearance` 5, `relation` 5, `multi_motion` 2, `temporal_order` 2,
`appearance` 1, `other` 1; 22 distinct videos; 12 cases with a disappearance
gap; 27 with absent frames; 80/80 with non-zero dilated overlap; 54 with ≥2
same-category distractors.

Target categories (pool, top-20): `person` 76, `sheep` 16, `zebra` 15,
`ape` 13, `monkey` 13, `earless_seal` 12, `cow` 12, `fish` 10, then `bird`,
`surfboard`, `tiger`, `duck`, `dog`, `chameleon`, `giraffe`, `hat` (6 each),
`jellyfish`, `knife` (5), `tennis_racket`, `sedan` (4).

## 4. Stable keys

Every case carries:

```
video_id, exp_id, query, target_obj_id, candidate_obj_ids,
same_category_obj_ids, annotated_frames, target_category, mining_signals{...}
```

`mining_signals` additionally carries `n_target_frames`, `n_frames`,
`query_type_scores`, `rank_score`, and `in_top_k`. These are the identifiers
used unchanged by M5.2 (annotation bundle) and M5.3 (evaluation).

## 5. Limitations (stated up front)

1. Only **53 videos** have locally-available instance GT reachable from the
   `valid` split, so the pool is not a uniform sample of Ref-YT-VOS; it is a
   sample of the subset with usable instance tracks.
2. `co_visible_ratio` uses **annotated** frames only. Annotated frames are
   every 5th raw frame, so short occlusions can fall between annotations.
3. `max_dilated_overlap` uses a fixed 2%-of-shorter-side dilation. It is a
   proximity proxy, not a measured occlusion event.
4. `query_type` is a keyword heuristic, not a parser; it should be read as a
   coarse stratification key only.
5. **Mining is an ambiguity *hypothesis generator*.** Nothing in this file
   claims a case *is* ambiguous for a human — that is exactly what the P2
   annotation stage is for.

## 6. Self-check of an earlier revision (why the pool was regenerated)

An earlier revision of the miner had two defects, both fixed before this
revision was produced:

* `co_visible_ratio` counted *pairs* rather than *frames*, so it could exceed 1
  (observed up to 3.95). It is now incremented once per frame where the target
  and ≥1 same-category distractor are jointly visible; the statistic is
  reported as ≤ 1 in the current `candidate_stats.json`.
* The intended inter-instance overlap signal used raw mask intersection, which
  is always 0 for Ref-YT-VOS because instance masks are disjoint by
  construction (it measured nothing). It is now `max_dilated_overlap`.

The earlier revision's `candidate_pool.json` is superseded; the committed pool
is the regenerated one.

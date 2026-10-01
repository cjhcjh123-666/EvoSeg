# Data audit

The capability set is fixed before training by sorting
`sha256("CPG-OVERFIT-42::video_id/expression_id")`.

- GroundMoRe: 32 official Sequential training expressions from 31 source videos.
- Long-RVOS: 32 official explicit-order training expressions from 24 source
  videos.
- No validation/test example is included.
- Subset manifests retain every source expression, object/annotation ID, source
  metadata checksum, and source root.
- One capability epoch is an exact deterministic interleave
  `Long[0], Ground[0], ..., Long[31], Ground[31]`; expression selection is not
  with replacement. Official VIRST random-within-bin frame sampling remains
  active, so repeated epochs expose the same expression to different legal
  frame samples without changing the fixed example pool.

GroundMoRe official train metadata contains 1,173 Sequential expressions from
414 videos. The deterministic single-connective parser resolves 1,112 and marks
61 unresolved. Exact counts and every source row are preserved in
`GROUNDMORE_QUERY_AUDIT.md`, `groundmore_query_audit.json`, and
`groundmore_query_templates.csv`.

Crucially, 442 groups reuse one `action_start/action_end` interval for questions
with different answer-bearing clauses, and 122 groups mix `before` and `after`
forms in the same interval. The interval remains valid for official temporal-mask
evaluation, but not for supervising one clause-specific process state. Unresolved
or interval-ambiguous records remain eligible for segmentation; they do not
receive a fabricated clause-localization target.

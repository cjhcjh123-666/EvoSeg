# GroundMoRe Sequential Query Audit

## Verified source

- Metadata: `/9950backfile/chenjiahui/evo_artifacts/datasets/GroundMoRe-official/trainval_v2.json`
- SHA-256: `df44c8ebd4b9ec6389bdde33aaf58fe176668357a4f231f162ec6fac402f904e`
- Sequential expressions: **1173** across **414** videos

## Deterministic language parsing

- Resolved by exactly one supported connective: **1112**
- Unresolved: **61**
- Connectives: `{"after": 666, "before": 443, "then": 3, "unresolved": 61}`

The parser only splits an explicit surface template and reorders `after` into
chronological order. It does not invent event labels or inspect masks.

## Interval semantics

The official metadata supplies one `action_start/action_end` pair per question.
There are **468** unique video/interval groups;
**461** are shared by multiple Sequential questions,
and **442** share the
same interval despite different answer-bearing clauses. **122**
groups contain both `before` and `after` formulations.

Therefore the interval is safe as the official mask-validity/full queried-process
window, but **not** as a clause-specific action annotation. CPG will preserve it
for official segmentation/evaluation. Clause-specific `L_loc` is disabled unless
an official clause-level field is found; using this interval as such would create
a pseudo-label.

## Representative shared intervals

- `4:15–4:23` (after): Who shoots the ball after dribbling the ball?
- `4:15–4:23` (after): What does the man in white shoot after dribbling?
- `4:15–4:23` (before): Who dribbles the ball before scoring a point?
- `4:15–4:23` (before): What does the man in white dribble before scoring a point?
- `4:58–5:03` (after): Who shoots the ball after dribbling it?
- `4:58–5:03` (after): What does the man in black shoot after dribbling?
- `5:09–5:12` (before): Who gets rid of the defense from the man in white before he shoots the basketball?
- `5:09–5:12` (before): Who does the man in black get rid of the defense from before he shoots the basketball?
- `5:17–5:22` (after): Who grabs the basketball after it is dropped?
- `5:17–5:22` (after): What does the man in black grab after he drops it?
- `5:17–5:25` (after): Who shoots the basketball after picking it up?
- `5:17–5:25` (after): What does the man in black shoot after he picks it up?

# ProcessVIRST data audit

No new query, event label, pseudo-label, or manual annotation was created.

## Long-RVOS

- Validation protocol is the frozen paired manifest from the preceding study:
  274 objects, 90 source videos, and 1,189 official expressions.
- The official baseline result has 1,189/1,189 successful expressions.
- The preregistered training pilot selects 64 source videos before expressions:
  626 official expressions and 183 mask tracks.
- The deterministic explicit-order filter
  `before|after|then|first|finally|followed by|subsequently` matches 86
  expressions from 32 of those training videos.
- The ProcessVIRST Long-RVOS validation pilot was not run after the earlier
  GroundMoRe gate failed. This is an early stop, not a missing result reported
  as zero.

## GroundMoRe

- Official Sequential trainval metadata: 414 videos and 1,173 expressions.
- Locally runnable: 413 videos and 1,171 expressions.
- One official clip, `LM-JY2eoFp8_1534_1546`, is absent locally, affecting two
  training expressions; it is logged rather than silently discarded.
- Pilot evaluation: 16 source videos and 47 official Sequential expressions,
  with no missing clip or expression.
- Full official baseline evaluation: 147 videos and 480/480 expressions.
- Evaluation uses exactly 20 uniformly sampled frames. Ground truth is zero
  outside the official 6 FPS action interval, and instance IDs are preserved.

All split construction and deterministic filtering happened before model
evaluation and did not use predictions.

# Ordered Process Grounding (OPG) SFT

OPG preserves the frozen order-agnostic multi-frame identity score and adds only a learned residual: `s_final = s_base + sigmoid(alpha) * s_order`. Four learned process slots cross-attend the untouched official query-token sequence, receive ordinal embeddings and validity gates, and align to eight ordered frozen candidate-region features through a differentiable log-sum-exp over all strictly increasing paths. The selected frozen SAM3.1 candidate track is emitted directly; the order branch never predicts pixels.

- Trainable OPG parameters: 1,289,218
- Frozen: Sa2VA, SAM3.1, candidate generator and feature encoders
- Order negative: exactly one identity-hashed reverse or half-block swap per verified sequential training sample
- Fixed loss: `L_id + L_order`, margin 0.2, lambda 1.0

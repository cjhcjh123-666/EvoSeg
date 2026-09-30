# Discriminative Ordered Process Grounding (OPG-v2)

OPG-v2 retains the four process slots, validity gates, frozen T=8 candidate features, and monotonic log-sum-exp alignment from v1. It adds candidate-order CE, candidate-relative order CE, and target self-order margin supervision. Candidate-wise standardized base, original-order, and original-minus-permutation logits are fused with positive learned `beta_order` and `beta_delta`, both initialized to 1. Frozen SAM3.1 candidate masks remain the spatial output.

No VLM, SAM3.1, candidate generator, or visual encoder parameter is updated. No artificial query, event label, GT inference prompt, RL, Agent, or CoT is used.

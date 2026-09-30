# ProcessVIRST SFT failure analysis

The failure is not a silent fusion gate. Beta stayed near 1.003 in every run,
more than 3.217 million trainable values changed, and original/permuted inputs
produce different pre-prompt residuals. Synthetic tests also confirm finite
gradients and order-sensitive monotonic DP behavior.

The observed failure is that the SFT objective did not learn a stable *correct*
order preference:

- the 0.2 hinge loss stayed near 0.2;
- seed 11 became slightly original-preferring;
- seed 23 remained near indifferent;
- seed 42 learned a statistically negative preference on the pilot;
- no-order and global controls reached essentially the same segmentation J&F.

Thus numerical sensitivity to a permutation did not become a reliable task
signal. The process residual altered prompts, but those changes neither tracked
the requested order consistently nor improved masks beyond the controls.

Representative cases are listed in `qualitative_order_cases/README.md`, with
positive, near-zero, wrong-direction, and zero-J&F examples. Alignment heatmaps
for three fixed cases are under `figures/qualitative_order_cases/`.

The pilot's fixed 384-update budget may be insufficient, but extending it now
would violate the preregistered early-stop rule and turn the final benchmark
into a tuning target. This run therefore supports NO-GO for the current SFT
configuration, not a universal impossibility claim about process-aware VIRST.

# Final ProcessVIRST SFT decision

Decision: **NO-GO**.

The implemented structure places ordered process reasoning before the official
VIRST SegPrompter and changes frame-specific prompt states, so the intended
integration point is real. The three FP32 seeds and both controls completed
successfully, with 3,217,666 trainable parameters and beta remaining near 1.003
rather than collapsing to zero.

However, the decisive behavioral requirements failed:

1. GroundMoRe pilot segmentation improved by only +0.088 pp on the three-seed
   arithmetic mean, CI [-0.061,+0.336] pp.
2. Full versus matched no-order was +0.025 pp, CI [-0.043,+0.111] pp.
3. Three-seed Original-Reverse margin was -0.00069, CI
   [-0.00222,+0.00094]; BlockSwap was -0.00002, CI
   [-0.00125,+0.00115].
4. Seed 11 preferred the original order slightly, seed 23 was near zero, and
   seed 42 significantly preferred the wrong order. Selecting seed 11 would be
   result-dependent seed selection and is prohibited.
5. The fixed 0.2 order-ranking loss stayed near 0.2, showing that the short SFT
   did not learn the requested preference despite finite gradients and a
   non-collapsed process residual.

Because order margin is effectively zero and normal-task improvement is tiny
and not attributable to order supervision, the pilot fails before the
Long-RVOS/full expansion gate. No RL design or training is started. Increasing
the epoch budget, changing lambdas, or choosing the favorable seed would be a
new experiment requiring separate preregistration; it is not performed here.

The supported conclusion is narrow: this ProcessVIRST SFT implementation makes
order structurally representable and changes pre-prompt states, but the fixed
pilot SFT does not teach a stable original-order preference or improve the
official segmentation task. It does not show that ordered pre-prompt reasoning
is impossible in general.

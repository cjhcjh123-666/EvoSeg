# FTSG method

Factorized Temporal-Spatial Grounding explicitly separates **which object** from **which pixels**. A frozen Sa2VA cumulative state and ordered candidate-region sequence feed the verified single-layer BiGRU identity selector. The selected frozen SAM3.1 concept candidate track is returned directly; temporal state never generates a heatmap or pixel prompt. Sa2VA, SAM3.1, and visual encoders remain frozen. GT is used only on official train to assign a candidate identity label and only at evaluation to score outputs.

Controls use the identical candidate bank and direct-track executor. Static identity sees only the final anchor appearance; order shuffle permutes only the four candidate-region features with a fixed per-expression derangement; mean pool removes order using a parameter-matched scorer.

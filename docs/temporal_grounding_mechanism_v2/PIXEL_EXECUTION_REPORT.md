# Pixel Execution Decomposition

The candidate identity and four-stage plan are fixed across C0/C1/C2 and C3/C4. Candidate masks generate prompts; GT is opened only after prompt inference. The matrix contains all **1100/1100 unique successful conditions**. It preserves **1 failed attempt(s)** separately; their keys were retried successfully and are counted once in the matrix. Point and box use the Meta public multiplex `add_prompt` wrapper. The public multiplex model has no stable mask-prompt method (`add_mask` explicitly rejects it), so MASK_INIT / MASK_PROPAGATE are **N/A**. Box prompting uses official semantic-box behavior, which resets semantic state at each stage; this limitation is part of the measured public interface.

| identity_basis | prompt_form | description_type | expressions | direct_track_J_and_F | anchor_candidate_J_and_F | init_J_and_F | candidate_to_init_IoU | propagated_J_and_F | initialization_loss_J_and_F | propagation_loss_J_and_F | direct_to_propagated_loss_J_and_F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| predicted_static | point | static | 98 | 61.07 | 65.40 | 26.17 | 37.50 | 46.26 | -39.23 | 20.09 | -14.81 |
| predicted_static | point | dynamic | 97 | 58.99 | 62.26 | 26.17 | 25.59 | 45.00 | -36.09 | 18.83 | -13.99 |
| predicted_static | point | hybrid | 80 | 56.50 | 59.62 | 24.58 | 38.26 | 41.02 | -35.03 | 16.44 | -15.48 |
| predicted_static | box | static | 98 | 61.07 | 65.37 | 47.03 | 70.94 | 26.63 | -18.33 | -20.40 | -34.44 |
| predicted_static | box | dynamic | 97 | 58.99 | 62.27 | 45.20 | 66.10 | 26.51 | -17.07 | -18.69 | -32.49 |
| predicted_static | box | hybrid | 80 | 56.50 | 59.61 | 43.82 | 70.12 | 24.71 | -15.79 | -19.11 | -31.79 |
| oracle_identity | point | static | 98 | 63.96 | 67.29 | 26.17 | 44.92 | 46.89 | -41.12 | 20.72 | -17.07 |
| oracle_identity | point | dynamic | 97 | 71.60 | 74.61 | 26.17 | 38.67 | 50.72 | -48.44 | 24.55 | -20.88 |
| oracle_identity | point | hybrid | 80 | 60.45 | 65.18 | 24.58 | 46.60 | 43.81 | -40.59 | 19.22 | -16.65 |
| oracle_identity | box | static | 98 | 63.96 | 67.26 | 46.92 | 72.99 | 26.63 | -20.34 | -20.29 | -37.33 |
| oracle_identity | box | dynamic | 97 | 71.60 | 74.61 | 50.88 | 71.07 | 27.08 | -23.72 | -23.80 | -44.52 |
| oracle_identity | box | hybrid | 80 | 60.45 | 65.17 | 45.59 | 73.77 | 24.91 | -19.58 | -20.68 | -35.54 |

`initialization_loss_J_and_F = init - anchor`; `propagation_loss_J_and_F = propagated - init`; and `direct_to_propagated_loss_J_and_F = propagated - C0 DIRECT_TRACK`. Negative values are losses. C1/C3 score only the immediate anchor output, whereas C2/C4 score the propagated video.

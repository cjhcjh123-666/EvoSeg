# Temporal controls

On Dynamic expressions:

- Normal temporal − fixed order shuffle selection accuracy: -2.15 pp, 95% CI [-4.71, +0.34].
- Normal temporal − fixed order shuffle J&F: -0.81 pp, 95% CI [-2.17, +0.34].
- Normal temporal − parameter-matched mean pool selection accuracy: -2.47 pp, 95% CI [-4.88, -0.14].
- Normal temporal − parameter-matched mean pool J&F: -0.79 pp, 95% CI [-1.74, +0.05].

Order shuffle keeps the candidate, frame set, spatial content, query, and cumulative Sa2VA states fixed; only the four candidate-region features are deterministically permuted. Mean pooling is trained on the same official-train split with seeds 11/23/42 and differs from the BiGRU only in its order-free aggregation/scorer.

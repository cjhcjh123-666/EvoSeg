# GroundMoRe Sequential results

## Official VIRST baseline

The official VIRST/SAM2 pipeline was evaluated directly, without a SAM3.1
candidate bank. The full 480-expression run completed with zero failures:

| Split | Expressions | J | F | J&F |
|---|---:|---:|---:|---:|
| pilot16 | 47 | 26.121 | 29.340 | 27.731 |
| full Sequential test | 480 | 25.543 | 27.967 | 26.755 |

The full baseline took 1:18:14, or 9.78 seconds per expression including video
decoding, VLM, prompt generation, and SAM2 propagation.

## ProcessVIRST pilot16

| Condition | Seed | J | F | J&F | Delta vs official (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Full monotonic + order loss | 11 | 26.231 | 29.521 | 27.876 | +0.145 | [-0.011, +0.463] |
| Full monotonic + order loss | 23 | 26.092 | 29.382 | 27.737 | +0.007 | [-0.100, +0.101] |
| Full monotonic + order loss | 42 | 26.203 | 29.481 | 27.842 | +0.112 | [-0.096, +0.457] |
| Monotonic, no order loss | 11 | 26.200 | 29.502 | 27.851 | +0.121 | [-0.090, +0.484] |
| Global/non-monotonic | 11 | 26.172 | 29.485 | 27.828 | +0.098 | [-0.112, +0.449] |

The three-seed arithmetic mean for Full ProcessVIRST is 27.818 J&F. Its paired
delta from official VIRST is +0.088 pp with source-video cluster-bootstrap 95%
CI [-0.061, +0.336] pp. This is neither stable nor distinguishable from zero.

At matched seed 11, Full minus no-order is only +0.025 pp, CI
[-0.043, +0.111] pp. Full minus global is +0.047 pp, CI
[-0.024, +0.143] pp. Consequently, the observed tiny segmentation change is not
attributable to monotonic order supervision.

The three Full runs averaged 8.48 seconds per expression under concurrent pilot
execution, versus 7.40 seconds for the separately run official pilot. This is
an observed run latency, not a controlled efficiency claim.

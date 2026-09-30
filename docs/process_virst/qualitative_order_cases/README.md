# Fixed qualitative audit cases

Cases are selected mechanically from the three-seed mean order margins, not by
manual preference. Positive, near-zero, wrong-direction, and segmentation
failure examples are all retained.

## Largest positive original-order preference

- `9KhJO4UloIE_0404_0414/7`: “What does the man in white throw the basketball
  off of before he scores?” Reverse +0.00567, BlockSwap +0.00338, J&F 10.00.
- `pFX4CC5UiOY_0915_0925/9`: “Who secures the ball after the man in the grey
  shirt fails to score a point?” +0.00329 / +0.00551, J&F 24.96.
- `9KhJO4UloIE_0700_0715/4`: “Around whom does the man wearing red pants spin
  before scoring a point?” +0.00421 / +0.00420, J&F 29.63.

These are not clean successes: even the largest positive preferences have weak
segmentation, illustrating why order response alone is insufficient.

## Bag-of-frames / near-zero preference

- `pFX4CC5UiOY_0832_0846/4`: -0.00043 / +0.00051, J&F 7.06.
- `pFX4CC5UiOY_0705_0715/4`: -0.00115 / +0.00131, J&F 4.38.
- `NomMansxnQM_0200_0215/3`: +0.00040 / -0.00079, J&F 34.44.

## Wrong-direction preference

- `NomMansxnQM_0252_0302/3`: -0.00803 / -0.00421, J&F 55.41.
- `9KhJO4UloIE_0434_0444/6`: -0.00558 / -0.00623, J&F 2.33.
- `9KhJO4UloIE_0434_0444/7`: -0.00634 / -0.00456, J&F 37.97.

## Segmentation failures

- `pFX4CC5UiOY_0705_0715/5`: J&F 0.00.
- `pFX4CC5UiOY_0743_0753/4`: J&F 0.00.
- `pFX4CC5UiOY_0915_0925/2`: J&F 0.00.

Static-shortcut cases cannot be drawn from this official Sequential-only pilot.
The planned Long-RVOS pilot that contained Static queries was not run after the
decision gate failed; such cases are therefore marked N/A rather than invented.

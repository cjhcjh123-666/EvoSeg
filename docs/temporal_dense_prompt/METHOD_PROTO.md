# TDSP method prototype

Frozen Sa2VA cumulative or anchor-only state is combined with the frozen official SAM3.1 256x72x72 propagation feature map by a 82,817-parameter dot-product dense head. K=4, threshold=0.5, and deterministic P1/P8 point conversion were fixed before validation. SAM3.1 and Sa2VA remained frozen; only official Long-RVOS train masks supervised BCE+Dice. Public `add_prompt` points update one persistent `obj_id`; no private mask-prompt API or validation tuning was used.

# projects/evoseg — EvoSeg research layer

Own additions on top of the Pixel-LLM/Sa2VA base.

```
evoseg/
├── tools/        # data manifest builders, anti-shortcut tooling, eval metrics (ported)
├── configs/      # training / eval configs (SFT + faithfulness GRPO)
└── datasets/     # faithful-seg data builders (swap / no-object / counterfactual / clarify)
```

## Faithful-seg components (planned)
1. `datasets/` — query-swap + no-object + counterfactual + clarification data builders.
2. `configs/` — Sa2VA-aligned SFT config (Qwen3-VL-4B/8B), faithfulness GRPO config.
3. `tools/` — paper-aligned eval (cIoU/gIoU/Acc@0.5/J&F), N-acc abstention metrics,
   video hallucination benchmark curation.

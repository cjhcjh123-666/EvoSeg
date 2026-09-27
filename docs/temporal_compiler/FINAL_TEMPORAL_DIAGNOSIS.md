# Final Temporal Diagnosis Gate

Gate date: 2026-09-27. This decision uses the completed Sa2VA representation run, three-model 274-object comparison, and SAM3.1 274-object candidate audit. It was made before inspecting matcher prototype outcomes.

## Q1. Dynamic gap 属于哪一类？

**Main decision: mainly weak/implicit or limited-temporal models（Case B，有限定）。**

Sa2VA 和 InstructSeg 的全量 Dynamic−Static J&F 分别为 −3.37pp（95% CI [−4.80,−1.90]）和 −4.73pp（[−6.38,−2.92]）；VIRST 为 −0.37pp（[−2.68,+1.83]）。VIRST 的强时序组织与输入协议伴随 gap 显著缩小，因此现象不是 Sa2VA-specific，也不是在所有家族中同等存在。

限定：InstructSeg 包含局部 temporal aggregation，模型之间也同时改变训练数据、输入帧、视觉编码器和 pixel executor。因此证据支持“VIRST 整体大幅缩小 gap”，不支持把效果单独归因给 STF、TDAU 或帧数。

## Q2. Sa2VA N=8→32 无明显收益主要支持什么？

**Grounding-interface bottleneck 是当前最强解释；pure representation bottleneck 不支持，temporal organization 仍可能参与。**

- Dynamic `z_seg` N8→N32 cosine 0.9805、normalized L2 0.1747；表示有变化，并非冻结不动。
- 该 movement 小于同对象 Static↔Dynamic@N16 的 0.2555，更远小于同视频不同对象的 0.6968。
- `z` distance 与 ΔJF 的 Spearman ρ=0.090，95% CI [−0.026,+0.199]，没有稳定关联。
- N8/N32 prediction mask IoU 0.7940，frame-wise disagreement 0.0188，而 Dynamic J&F 只变 +0.392pp，95% CI 跨零。

因此，更多帧确实改变 VLM/segmentation token，但单个全局 `z_seg` 到 pixel executor 的接口没有把变化稳定转化成正确目标轨迹。现有观测不能严格区分“接口压缩”与更早的“时序组织不足”，故后者保留为次要、待消融解释。

## Q3. 显式 temporal VLM 是否显著缩小 gap？

**是，就本次跨模型观测而言 VIRST 显著缩小了模型内 gap。** VIRST 的 gap 绝对值比 Sa2VA 小 3.00pp、比 InstructSeg 小 4.36pp，并且自身区间跨零。不过这不是同 checkpoint 的结构消融，不能写成单模块因果证明；它是 Method 方向和 motivation 的跨模型证据。

## Q4. SAM3.1 candidate bank coverage 是否足够？

**足够做有限的最小 matcher prototype，但不够视为已解决。** deterministic concept 在 272 个完整 Dynamic objects 上 oracle J&F 0.6892，Recall@0.3/0.5/0.7 为 91.73%/81.62%/58.46%。3,567 个条件中 16 个官方 tracker 失败均保留；candidate miss 必须在端到端测试中记零，训练时不强配错误正样本。

raw official expression 的 Dynamic Recall@0.5 仅 34.19%，因此 Object Multiplex 的输入必须是 deterministic object concept；完整 official expression 只用于 query-conditioned track selection。

## Method route

**选择 Route B，并吸收 Route D 的接口设计：compact Temporal Compiler + SAM3.1 Candidate Track Bank + query-conditioned Temporal Track Matcher。**

理由是：强时序 VIRST 已大幅缩小 gap，但其原生协议使用多达 64 个 VLM 帧、32 个 SAM2 prompt 帧和全视频传播；Sa2VA 的全局表示变化没有稳定穿过 segmentation interface；SAM3.1 concept candidate bank 又为 81.62% 的 Dynamic objects 提供了可选真值轨迹。最小实验因此检验：在冻结 VLM、视觉塔和 SAM3.1 时，保留有序 track features 的 parameter-matched scorer 是否优于先做时间均值的 static scorer。

Prototype gate 通过，但边界如下：

- 只训练 2-layer lightweight matcher；不训练/微调 SAM3.1、Sa2VA 或视觉塔。
- source-video-disjoint train/val/test；固定 split seed=42，三个模型种子。
- static/temporal scorer 参数完全相同，区别只有时间顺序是否保留。
- 只用官方表达和 GT mask 匹配 candidate target；不生成 query 或伪标签。
- candidate miss 独立保留。若 Dynamic 上 temporal ≤ static 或区间跨零，则停止，不 claim temporal sequence modeling 有效，也不进入更大训练。

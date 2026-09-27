# Dynamic Grounding Interface results

本页只统计七个条件均成功的完整 expression；失败记录保留且不补值。当前覆盖 64 videos、64 objects、275 expressions，terminal 失败 0。

275 条官方 expression 包含 Static 98、Dynamic 97、Hybrid 80。七条件共有
1925 个唯一终态键，全部成功；对象先按同类型 expression 平均，再等权统计
64 个对象（本 pilot 每个对象来自不同 source video）。原始 artifact 位于
`/9950backfile/chenjiahui/evo_artifacts/results/temporal_grounding_interface/20260927_tgi_pilot64_v2/`。

## 主结果

| Condition | Type | J | F | J&F |
|---|---:|---:|---:|---:|
| single_global | static | 24.37 | 24.37 | 24.37 |
| single_global | dynamic | 24.37 | 24.37 | 24.37 |
| single_global | hybrid | 22.60 | 22.60 | 22.60 |
| static_update_k2 | static | 34.04 | 33.84 | 33.94 |
| static_update_k2 | dynamic | 34.84 | 35.14 | 34.99 |
| static_update_k2 | hybrid | 29.77 | 30.23 | 30.00 |
| static_update_k4 | static | 43.34 | 43.09 | 43.21 |
| static_update_k4 | dynamic | 43.34 | 43.54 | 43.44 |
| static_update_k4 | hybrid | 38.38 | 38.28 | 38.33 |
| static_update_k8 | static | 49.78 | 49.65 | 49.71 |
| static_update_k8 | dynamic | 49.55 | 49.62 | 49.59 |
| static_update_k8 | hybrid | 46.35 | 46.30 | 46.32 |
| temporal_update_k2 | static | 33.37 | 33.19 | 33.28 |
| temporal_update_k2 | dynamic | 34.27 | 35.09 | 34.68 |
| temporal_update_k2 | hybrid | 30.17 | 30.67 | 30.42 |
| temporal_update_k4 | static | 43.33 | 43.04 | 43.19 |
| temporal_update_k4 | dynamic | 43.53 | 44.02 | 43.78 |
| temporal_update_k4 | hybrid | 38.31 | 38.37 | 38.34 |
| temporal_update_k8 | static | 48.61 | 48.30 | 48.45 |
| temporal_update_k8 | dynamic | 50.94 | 50.99 | 50.96 |
| temporal_update_k8 | hybrid | 44.38 | 44.29 | 44.33 |

预注册主比较是 K=4 的 Dynamic object mean：Temporal - Static = 0.34 pp，source-video cluster bootstrap 95% CI [-1.49, 2.48] pp。

Single-global Dynamic-Static gap = 0.00 pp；Temporal K4 gap = 0.59 pp；绝对 gap 缩小 -0.59 pp。

这里不能把 Single-global 的 0.00pp 当成“消除 Dynamic gap”：其 275/275
expression 的保存 mask 全空，24.37 J&F 来自空 GT 帧对空预测的计分。它是
最后 stage 初始化再向全片反向传播这一具体接口协议的失败，而非有效性能基线。

K4 Dynamic 中，Temporal 与 Static 产生不同 candidate sequence 的 expression 比例为 21.65%。该诊断区分“表示变化没有越过离散 grounding 决策边界”和“改变了 referent 但 tracking/掩码未改善”。

K4 Dynamic 的 initial-wrong 后续 correction rate 为 Temporal 67.31%、Static
66.67%，仅高 0.64pp；Temporal 被纠正的 35 条 Dynamic expression 上，
Temporal-Static J&F 平均为 -1.08pp。与此同时，initial-correct 后被破坏的比例
是 Temporal 60.00%、Static 58.14%。因此 correction 计数没有转化为稳定像素收益。

## 运行代价

| Condition | Dynamic median total (s) | VLM forwards | Updates | Peak GiB |
|---|---:|---:|---:|---:|
| single_global | 67.85 | 1.00 | 0.64 | 15.71 |
| static_update_k2 | 72.74 | 2.00 | 1.30 | 71.63 |
| static_update_k4 | 108.81 | 4.00 | 2.68 | 72.32 |
| static_update_k8 | 194.99 | 8.00 | 5.35 | 73.68 |
| temporal_update_k2 | 71.02 | 2.00 | 1.30 | 71.63 |
| temporal_update_k4 | 109.42 | 4.00 | 2.68 | 72.32 |
| temporal_update_k8 | 194.62 | 8.00 | 5.35 | 73.68 |

延迟均包含 CUDA synchronize；tracking cache hit 仍报告首次独立执行该 trajectory 的同步延迟。VLM/candidate 缓存来自前置独立运行，且部分 GPU 与既有任务并行，因此这些数字用于如实记录本轮代价，不作为隔离效率结论。

## Decision gate

- A（K4 Temporal > Static 且三档至少两档方向为正）：True
- B（相对 Single-global，gap 至少缩小 1pp）：False
- C（Dynamic correction rate 高于 Static control，且被纠正样本的 Temporal-Static J&F 为正）：False；被纠正 35 条，平均关联增益 -1.08 pp
- Single-global 输出活动性检查：False（逐条件明细见 `mask_activity_summary.csv`；全空 mask 不会因 GT 缺席帧的 J/F=1 被误判为有效输出）
- **NO-GO**

这只是当前冻结 Sa2VA + 官方 SAM3.1 point refinement、零训练参数 scorer 的结论；不外推为所有动态接口均有效或无效。

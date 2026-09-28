# Final Temporal Grounding Mechanism Diagnosis

## Decision

**Route D — Mixed。**

同时保留一个更具体的机制判断：在已测链路中，最大的可操作损失发生在 `candidate track → single point prompt → SAM3.1 tracking`，但 candidate generation ceiling 和 object selection 也都不可忽略，因此不把结论强行压成纯 Route B。

## 证据合并

1. **冻结 temporal signal 可以被轻量 probe 读出，但 pilot 不够精确。** Dynamic selection accuracy 从 57.76% 提升到 65.52%；candidate-direct J&F 从 51.72 提升到 55.45，差值 `+3.73 pp`，source-video bootstrap 95% CI `[-0.85, +9.03] pp`。三个固定 seed 的 J&F 点估计均为正（+7.03/+3.93/+3.11 pp），所以不是挑 seed 得到的方向；但 ensemble CI 跨 0，不能宣称强显著。

2. **Sa2VA 多帧 representation movement 很少改变 candidate ranking。** N=8/16/32 的 Dynamic top-1 为 76.29%/78.35%/77.32%；N8→N32 只有 5.15% expression 改变 top-1，margin 变化 −0.00075。representation distance 与 margin change 的 ρ=0.213，95% CI [-0.011, 0.425]；与 J&F change 的 ρ=0.052，CI [-0.197, 0.288]。

3. **当前无训练 selector 有 selection loss。** Dynamic ORACLE candidate-direct 71.60，而 static predicted candidate-direct 为 58.99，差 12.60 pp。当前 temporal selector direct 为 58.04，并未利用好可读 temporal signal。

4. **point/tracking 丢失更大。** static predicted candidate 从 direct 58.99 降为 point/tracking 45.00（−13.99 pp）；即使 ORACLE 选择正确 track，同一 pixel interface 也只有 50.72，相对 oracle track 自身损失 20.88 pp。

5. **candidate bank 有余量但仍有限制。** Dynamic oracle track J&F 71.60，明显高于当前最终路径，但距完美仍有 28.40 pp；它不是当前唯一瓶颈，也不能视作已解决。

## 对预设 Gate 的回答

- “时序信息存在且可读”获得**有限支持**：三 seed 同向、candidate-direct 改善，但 pilot CI 跨 0。
- “当前 grounding interface 没有充分利用”获得支持：旧 temporal selector 的 direct/point 没有稳定收益，且 point/tracking 对 predicted 与 ORACLE identity 都产生大幅下降。
- 由于 selection、pixel execution、candidate ceiling 三处都有明显损失，最终不选纯 Representation、纯 Interface 或纯 Candidate 路线，选择 Mixed。

## 下一步建议

不训练复杂 Method。下一步只做一个受控接口实验：固定 diagnostic probe 选出的同一 candidate，比较 `candidate-direct mask`、官方 mask/box prompt（若公开稳定 API 支持）与当前 single-point prompt 的同帧初始化及传播，严格使用相同 track、相同评价帧。目标是把 20.88 pp 的 combined point/tracking gap 分解为 prompt information compression 与 video propagation 两部分。若 mask/box prompt 仍无法保留 direct-track 收益，再检查 pixel executor；若能保留，再讨论 temporal state 到 dense pixel prompt 的接口。candidate-miss 样本继续独立报告，不进入错误正标签。

本轮到此停止，不启动最终 Method 或更大规模训练。

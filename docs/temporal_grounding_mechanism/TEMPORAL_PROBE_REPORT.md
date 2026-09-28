# Frozen diagnostic probe：时序信息能否被读出

## 协议

- Sa2VA、SAM3.1 candidate bank 和 InstructSeg/SigLIP region encoder 全部冻结。
- 训练只使用 Long-RVOS official train：seed 42 按源视频抽取 64 个视频/对象、312 条官方表达，保留官方 `type` 字段；与 pilot validation 的源视频交集为 0。
- candidate bank 的 oracle J&F 小于 0.3 或为空时记为 `candidate_miss`，不制造正标签、不进入 CE；这些样本仍进入 candidate-direct J&F，得分为 0。
- official train 中 280/312 expressions 有可训练 candidate；按源视频划分为 228 条 fit、52 条 early-stop validation。
- Static 与 Temporal 使用完全相同的单层 BiGRU scorer，各 288,257 个可训练参数。Static 只读 final anchor region feature 与 `static_7_z`，并重复为四步；Temporal 读四个有序 region features 与 cumulative Sa2VA `temporal_{1,3,5,7}_z`。
- 固定 seeds 为 11/23/42，不按 validation pilot 选择 seed；主结果为三个模型 logits 的平均。pilot 共 64 个对象、275 条官方表达。

## Pilot-64 结果

对象内同类型多表达先平均，再对对象平均。Selection accuracy 只在 candidate hit（oracle J&F ≥ 0.3）上定义；candidate-direct J&F 包含 candidate miss。

| 描述类型 | Probe | hit expressions / objects | selection accuracy | candidate-direct J&F |
|---|---|---:|---:|---:|
| Static | Static | 81 / 56 | 63.39% | 48.04 |
| Static | Temporal | 81 / 56 | 68.75% | 52.94 |
| Dynamic | Static | 89 / 58 | 57.76% | 51.72 |
| Dynamic | Temporal | 89 / 58 | 65.52% | 55.45 |
| Hybrid | Static | 64 / 48 | 67.71% | 45.53 |
| Hybrid | Temporal | 64 / 48 | 70.83% | 47.78 |

Dynamic 的 Temporal − Static：

- selection accuracy：`+7.76 pp`，source-video cluster bootstrap 95% CI `[-3.45, +18.97] pp`；
- candidate-direct J&F：`+3.73 pp`，95% CI `[-0.85, +9.03] pp`。

## Seed 稳定性

没有挑 seed。三个预注册 seed 的 Dynamic candidate-direct 增益分别为 `+7.03`、`+3.93`、`+3.11 pp`；selection accuracy 增益分别为 `+12.07`、`+7.76`、`+6.90 pp`。三个点估计均同向，但除 seed 11 的 candidate-direct J&F 外，单 seed CI 仍跨 0；ensemble CI 也跨 0。

## 解释边界

最小、参数匹配的 Temporal probe 能从冻结的多时点 region features 与 cumulative Sa2VA states 中读出比单 anchor control 更好的对象排序，且三个固定 seed 方向一致。这支持“可读的 object-discriminative temporal signal 并非完全缺失”。效应在 64-video pilot 上仍不精确，不能写成强显著性，也不能单独证明收益来自 `z_seg` 而非有序 region feature；本实验诊断的是两者组成的最小 temporal readout。

逐表达结果见 `probe_results.csv`，训练曲线、环境、split、三 seed 与 2000 次 bootstrap 明细保存在 artifact 的 `probe/probe_audit.json`。

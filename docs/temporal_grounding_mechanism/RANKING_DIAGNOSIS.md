# Sa2VA 跨帧预算 candidate ranking 诊断

## 范围与定义

- 数据：与 TGI pilot 完全相同的前 64 个 paired objects，共 97 条官方 Dynamic expressions、64 个源视频。
- 帧预算：严格复用上一轮 Sa2VA 的 N=8/16/32 输入帧、预测 mask 和 `z_seg`；没有重新分割。
- Candidate：复用冻结的 SAM3.1 concept candidate bank。GT 只用于事后定义 `oracle_correct_track_id`。
- Ranking score：Sa2VA 输出 mask 与每条固定 candidate track 在相同评价帧上的 mean IoU。它是一个可复核的、无训练的 operational grounding score，不等同于 diagnostic probe 的 learned score。
- 1 条 Dynamic expression 的 candidate bank 为空；其 top-1 计为失败，rank 记为缺失。其余 96 条用于 mean rank。

## 逐预算结果

| VLM 可见帧 | correct top-1 | mean correct rank | target-best distractor margin | Sa2VA J&F |
|---:|---:|---:|---:|---:|
| 8 | 76.29% | 1.708 | 0.29943 | 46.01 |
| 16 | 78.35% | 1.573 | 0.30116 | 46.14 |
| 32 | 77.32% | 1.667 | 0.29868 | 46.06 |

N=8→32 时，top-1 identity 仅有 5.15% 的 expressions 发生变化；平均 margin 变化为 `-0.00075`，平均 J&F 变化为 `+0.00047`（即 `+0.047 pp`）。增加四倍可见帧没有形成单调或持续的 target-vs-distractor margin 增益。

## 关联分析

按 source video 做 2000 次 cluster bootstrap：

| 变量 | Spearman ρ | 95% CI |
|---|---:|---:|
| representation cosine distance vs margin change | 0.213 | [-0.011, 0.425] |
| representation cosine distance vs J&F change | 0.052 | [-0.197, 0.288] |
| margin change vs J&F change | 0.422 | [0.164, 0.651] |

## 解释边界

结果符合“`z_seg` 会随帧预算变化，但大部分变化没有转成对象判别 margin”的描述。margin 一旦变化则与 J&F 变化有中等正相关，但 representation movement 本身既不能稳定预测 margin，也不能预测 J&F。该结果支持定位问题，但不证明 `z_seg` 完全没有可读 temporal information；后者由独立、严格 train/validation 隔离的 frozen diagnostic probe 检验。

原始轻量结果见 `ranking_diagnosis.csv`；增量与 bootstrap 明细保存在 artifact 目录的 `ranking_deltas.csv` 和 `ranking_correlations.json`。

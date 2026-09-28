# ORACLE Ladder：时序 grounding 链路损失定位

## 范围和防泄漏

- 数据固定为 TGI pilot 的 64 个 Long-RVOS validation videos/objects、275 条官方表达（Static 98、Dynamic 97、Hybrid 80）。
- SAM3.1 concept candidate、Sa2VA static/temporal stage states、评价帧和 GT 均复用冻结缓存；没有重跑候选生成或 Sa2VA segmentation。
- 所有 `ORACLE_*` 只在 candidate 已生成后读取 GT，用于事后选择/评价。GT mask、GT object id 均未进入 SAM3.1 prompt 或 tracking。
- 每种描述在对象内先平均，再对对象等权；本 pilot 每个对象来自不同 source video。
- 825 个 point 条件全部成功。运行中保留了 4 次双进程显存峰值导致的 OOM attempt，降为每卡单进程后 4 个键全部恢复；最终缺失为 0。

## 完整结果

| Condition | Type | J | F | J&F |
|---|---|---:|---:|---:|
| **ORACLE_CANDIDATE** | Static | 64.11 | 63.81 | 63.96 |
| **ORACLE_CANDIDATE** | Dynamic | 71.78 | 71.41 | 71.60 |
| **ORACLE_CANDIDATE** | Hybrid | 60.38 | 60.53 | 60.45 |
| PREDICTED_CANDIDATE_DIRECT_STATIC | Static | 61.00 | 61.15 | 61.07 |
| PREDICTED_CANDIDATE_DIRECT_STATIC | Dynamic | 58.39 | 59.60 | 58.99 |
| PREDICTED_CANDIDATE_DIRECT_STATIC | Hybrid | 56.29 | 56.71 | 56.50 |
| PREDICTED_CANDIDATE_DIRECT_TEMPORAL | Static | 59.28 | 59.37 | 59.33 |
| PREDICTED_CANDIDATE_DIRECT_TEMPORAL | Dynamic | 57.21 | 58.88 | 58.04 |
| PREDICTED_CANDIDATE_DIRECT_TEMPORAL | Hybrid | 54.63 | 55.20 | 54.91 |
| PREDICTED_CANDIDATE_POINT_STATIC | Static | 46.50 | 46.03 | 46.26 |
| PREDICTED_CANDIDATE_POINT_STATIC | Dynamic | 44.91 | 45.09 | 45.00 |
| PREDICTED_CANDIDATE_POINT_STATIC | Hybrid | 41.25 | 40.80 | 41.02 |
| PREDICTED_CANDIDATE_POINT_TEMPORAL | Static | 44.63 | 44.17 | 44.40 |
| PREDICTED_CANDIDATE_POINT_TEMPORAL | Dynamic | 45.04 | 45.63 | 45.33 |
| PREDICTED_CANDIDATE_POINT_TEMPORAL | Hybrid | 40.73 | 40.57 | 40.65 |
| **ORACLE_ID_POINT** | Static | 47.18 | 46.60 | 46.89 |
| **ORACLE_ID_POINT** | Dynamic | 50.99 | 50.45 | 50.72 |
| **ORACLE_ID_POINT** | Hybrid | 44.04 | 43.58 | 43.81 |

## Dynamic 损失阶梯

以 Static selector 作为当前 predicted path：

- `candidate_generation_gap = 100 − ORACLE_CANDIDATE = 28.40 pp`；这是相对完美 track 的 candidate-bank ceiling gap。
- `selection_gap = ORACLE_CANDIDATE − PREDICTED_CANDIDATE_DIRECT_STATIC = 12.60 pp`。
- `prompt_conversion_gap = PREDICTED_CANDIDATE_DIRECT_STATIC − PREDICTED_CANDIDATE_POINT_STATIC = 13.99 pp`。
- `tracking_gap = ORACLE_CANDIDATE − ORACLE_ID_POINT = 20.88 pp`。

Temporal selector 对应的 direct→point 差为 `12.71 pp`。`prompt_conversion_gap` 与 `tracking_gap` 是两个受控对照，不可相加：当前公开 API 条件把 candidate mask 压为 interior point 后重新 tracking，无法把“点转换”与“后续传播”再单独识别。

## 结论边界

candidate bank 并非无上限：Dynamic ORACLE_CANDIDATE 为 71.60，上一轮 Recall@J&F≥0.5 为 86.60%。但它也不是完美 candidate generator。对象选择造成约 12.60 pp 损失；即使 GT 只负责选对 candidate，point/tracking 路径仍从 71.60 降到 50.72，说明 pixel execution 是最大的已测链路内损失。多个阶段均有实质损失，因此不能把全部问题归因于单一模块。

原始逐表达 ladder 见 `oracle_ladder.csv`；`ORACLE` 列明确标出所有 GT-assisted 条件。

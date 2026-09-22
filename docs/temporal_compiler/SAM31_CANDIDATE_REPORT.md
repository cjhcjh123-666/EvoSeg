# SAM3.1 Candidate Track Bank 可行性：32-object pilot

## 结论

32 个预测无关抽样对象的 pilot 证明：**直接把完整官方表达交给 SAM3.1 并不能形成可靠 candidate bank；先抽取对象概念后，Dynamic target 的 oracle coverage 明显提高，达到可继续扩展审计的水平。** deterministic concept 的对象加权 Dynamic Oracle J&F 为 0.7160、Recall@J&F≥0.5 为 0.8594；Qwen text-only concept 分别为 0.7050 和 0.8438。相比之下，raw expression 只有 0.2479 和 0.2760。

这只是 32-object pilot 的 **candidate generation / oracle coverage**，不是 query-conditioned track selection 的最终分割分数，也不能证明 Dynamic query 已被解决。当前 decision 是放行至全部 274 paired objects 的 coverage 审计；在全量结果出来前，不训练 Temporal Matcher。

## 协议与完整性

- 源 manifest：上一轮固定 seed=42、同一 Long-RVOS paired-object manifest；pilot 取其预测无关顺序中的前 32 个对象，覆盖 32 个源视频。
- prompts：未改写的官方 raw expression、spaCy deterministic noun phrase、独立 Qwen text-only concept。parser 均不读取 target ID 或 mask。
- candidate generation：Meta 官方 SAM3.1 Object Multiplex，官方 repo commit `2345a4ad109ac29c569da749c91d84f10dc08c40`；checkpoint SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- GT 隔离：395 条成功生成记录均声明 `gt_read_during_generation=false`；GT 只在独立 evaluation 命令中用于选择 oracle-best candidate 和计算官方 J/F。
- 计划 396 个 prompt-expression 条件，成功 395，缺失成功条件 1。共保留 397 次生成尝试：395 成功、2 次失败尝试、1 次 retry；成功键无重复。
- 唯一失败键为 `long_rvos/6fb7a04167/12/25/qwen_concept`。官方表达为 Hybrid，Qwen concept 为 `man in a black coat`；原始尝试和一次重试都在官方 tracker 内以 `No points are provided; please add points first` 失败。同一表达的 raw 和 deterministic concept 条件成功。失败未删除，因此 Qwen Hybrid 汇总为 39 expressions / 30 objects；该 pilot 中有 Hybrid 表达的原始范围是 40 expressions / 31 objects。

## 对象加权结果

同一对象、同一描述类型的多条表达先平均，因此表达较多的对象不获得额外统计权重。

| Prompt | 类型 | 对象/表达 | Oracle J | Oracle F | Oracle J&F | Recall≥0.3 | Recall≥0.5 | Recall≥0.7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic concept | Static | 32/47 | 0.6123 | 0.6154 | 0.6139 | 0.7656 | 0.7188 | 0.5312 |
| deterministic concept | Dynamic | 32/45 | 0.7141 | 0.7180 | 0.7160 | 0.9062 | 0.8594 | 0.6562 |
| deterministic concept | Hybrid | 31/40 | 0.5606 | 0.5632 | 0.5619 | 0.6935 | 0.6613 | 0.5215 |
| Qwen concept | Static | 32/47 | 0.6336 | 0.6359 | 0.6348 | 0.8281 | 0.7500 | 0.5312 |
| Qwen concept | Dynamic | 32/45 | 0.7029 | 0.7071 | 0.7050 | 0.9062 | 0.8438 | 0.6250 |
| Qwen concept | Hybrid | 30/39 | 0.6682 | 0.6692 | 0.6687 | 0.8333 | 0.8000 | 0.5833 |
| raw expression | Static | 32/47 | 0.3961 | 0.3891 | 0.3926 | 0.5625 | 0.4375 | 0.2240 |
| raw expression | Dynamic | 32/45 | 0.2453 | 0.2504 | 0.2479 | 0.3542 | 0.2760 | 0.2135 |
| raw expression | Hybrid | 31/40 | 0.3298 | 0.3368 | 0.3333 | 0.4624 | 0.3656 | 0.2419 |

raw expression 在 132 条成功记录中有 60 条零候选；deterministic concept 为 13/132，Qwen concept 为 3/131。这个差异支持把 concept extraction 作为 candidate generation 的必要前端，而不是把完整关系/过程句直接当成 Object Multiplex 的对象概念。

## Candidate count

| Prompt | 类型 | mean | median | p95 | min–max |
|---|---|---:|---:|---:|---:|
| deterministic concept | Static | 5.87 | 4.50 | 14.45 | 0–16 |
| deterministic concept | Dynamic | 6.69 | 5.25 | 14.45 | 2–15 |
| deterministic concept | Hybrid | 5.04 | 3.00 | 15.00 | 0–16 |
| Qwen concept | Static | 6.14 | 4.25 | 15.00 | 0–16 |
| Qwen concept | Dynamic | 6.48 | 5.00 | 14.45 | 2–15 |
| Qwen concept | Hybrid | 5.07 | 4.00 | 11.55 | 1–14 |
| raw expression | Static | 1.54 | 1.00 | 4.00 | 0–7 |
| raw expression | Dynamic | 1.57 | 0.42 | 5.73 | 0–11 |
| raw expression | Hybrid | 1.91 | 1.00 | 7.00 | 0–7 |

## Decision

- **Pilot interface check：通过。** 官方 checkpoint、Object Multiplex、多候选 track、RLE 保存、GT-separated oracle evaluation 均真实跑通。
- **扩展 gate：通过但有限定。** concept prompts 在 Dynamic 上达到约 84%–86% Recall@0.5，足以继续全部 274 paired objects 的 coverage 测量；raw expression 明确不够。
- **Matcher training gate：尚未通过。** Static/Hybrid coverage 仍有明显缺失，pilot 样本也小；须先看全量 object-weighted coverage、失败分布和 candidate-miss 对象。
- oracle Dynamic 高于 Static 不能解释为模型已经理解动态描述：concept prompt 已丢弃大部分动态关系，oracle 又使用 GT 选择 candidate。它只说明目标 track 常在候选集合中。

## 随附轻量结果

- `results/sam31_candidate_pilot32_summary.csv`
- `results/sam31_candidate_pilot32_metrics.csv`
- `results/sam31_candidate_pilot32_evaluation_status.json`

原始候选 RLE、视频、GT 和模型权重只保存在 artifact 根目录，不进入 git。

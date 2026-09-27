# SAM3.1 Candidate Track Bank 可行性

## 全量主结果

全部 274 个 paired objects 的 candidate generation 已结束。计划且实际尝试 3,567 个 prompt-expression 条件，成功 3,551，失败 16，未尝试 0；失败全部保留。以下只在某 prompt/description cell 的全部官方表达成功时纳入 object-weighted 汇总，不用部分对象替代完整对象。

| Prompt | 类型 | 完整对象/表达 | Oracle J&F | Recall≥0.3 | Recall≥0.5 | Recall≥0.7 | candidates mean / median / p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| deterministic concept | Static | 273/423 | 0.6781 | 0.8980 | 0.7991 | 0.5739 | 6.84 / 5.5 / 16 |
| deterministic concept | Dynamic | 272/414 | **0.6892** | **0.9173** | **0.8162** | **0.5846** | 7.21 / 6 / 16 |
| deterministic concept | Hybrid | 262/344 | 0.6331 | 0.8480 | 0.7583 | 0.5204 | 6.00 / 4 / 16 |
| Qwen concept | Static | 273/423 | 0.6781 | 0.9121 | 0.8150 | 0.5531 | 7.11 / 6 / 16 |
| Qwen concept | Dynamic | 272/414 | 0.6748 | 0.9075 | 0.8045 | 0.5362 | 7.26 / 6 / 16 |
| Qwen concept | Hybrid | 260/342 | 0.6602 | 0.8846 | 0.7904 | 0.5212 | 6.51 / 4.25 / 16 |
| raw expression | Static | 273/423 | 0.4021 | 0.5501 | 0.4640 | 0.3016 | 2.41 / 1 / 9 |
| raw expression | Dynamic | 271/413 | **0.2996** | **0.4237** | **0.3419** | **0.2042** | 1.94 / 0.5 / 9 |
| raw expression | Hybrid | 262/344 | 0.3232 | 0.4612 | 0.3658 | 0.2201 | 1.93 / 1 / 7 |

deterministic concept 的 Dynamic Recall@0.5 为 81.62%，说明 candidate bank 对最小 matcher prototype **覆盖足够但并不完整**。约 18.38% 的完整 Dynamic 对象在该阈值仍 miss，必须独立记为 candidate miss，不能强行指定错误正样本。raw expression 的 34.19% 明显不足，说明完整过程句不是 Object Multiplex 的可靠对象概念 prompt。

## 生成完整性与失败

- 真实目录：`/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260922_sam31_candidate_all274`
- 四分片分别为 899/903、837/843、903/906、912/915 成功，状态均为 `complete_with_failures`。
- 16 个失败全部来自官方 tracker 的 `No points are provided` 路径；无 never-attempted key，无静默丢弃。
- deterministic concept：Static 273/274 完整、Dynamic 272/274、Hybrid 262/263。
- Qwen concept：Static 273/274、Dynamic 272/274、Hybrid 260/263。
- raw expression：Static 273/274、Dynamic 271/274、Hybrid 262/263。
- candidate generation 只读视频和 prompt；GT 仅在独立 evaluation 阶段选择 oracle-best track。

## Decision

- **Candidate-generation gate：有条件通过。** deterministic concept 是首选，Dynamic coverage 足以检验轻量 matcher，但不是部署级完整召回。
- **允许的训练范围：** 冻结 SAM3.1、VLM 和视觉塔，只训练最小 parameter-matched static/temporal track scorer；candidate miss 不进入正样本训练，并在端到端测试中保留为 0。
- **不允许的结论：** oracle 高不代表 query-conditioned selection 已解决，也不代表 SAM3.1 理解了 dynamic expression；oracle 使用 GT，仅测候选集合覆盖。
- 若 temporal scorer 未在 Dynamic 上优于 parameter-matched static scorer，则不能 claim temporal sequence modeling 有效，并在最小 prototype 后停止。

## 32-object pilot（保留）

pilot 覆盖 32 个源视频：deterministic concept 的 Dynamic Oracle J&F / Recall@0.5 为 0.7160 / 0.8594，Qwen concept 为 0.7050 / 0.8438，raw expression 为 0.2479 / 0.2760。全量保持了 concept 明显优于 raw 的方向，但数值有所回落。

## 轻量结果

- `results/sam31_candidate_all274_summary.csv`
- `results/sam31_candidate_all274_completeness.csv`
- `results/sam31_candidate_all274_evaluation_status.json`
- `figures/sam31_candidate_all274_coverage.png`

3MB 逐表达 metrics、原始候选 RLE、视频、GT 和 checkpoint 仅保存在 artifact 根目录，不进入 git。

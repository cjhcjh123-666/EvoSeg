# Failure analysis

## 完整性与运行失败

- 1925/1925 个唯一 `(identity, condition)` 终态成功，终态失败 0，不完整
  expression 0；所有 64 objects 和 275 expressions 进入主要统计。
- 原始日志保留 1 次历史 `failed_oom`：GPU1 与既有进程并行时，
  `long_rvos/39db436191/3/4, static_update_k4` 分配额外 42MiB 失败。它随后在
  独占 GPU3 上成功重试；终态去重没有静默删除该次尝试。
- 为加速尾段产生 8 个重复成功尝试；最终按 `(identity, condition)` 取最后一条，
  并验证每个条件恰有 275 个成功键。
- 早期 API lifecycle smoke 在首个 object point 前调用 propagation 会失败；修复为
  “尚未建立 track 时不传播”。这些无效 smoke 目录未纳入正式汇总。

## Single-global 退化

Single-global 的 275/275 expression、17600/17600 evaluation frames 全为空
mask；118 条没有末 stage candidate，另 157 条虽然调用了 point 初始化仍产生空
输出。独立复现样本中 grounding mask 与所选 candidate IoU=0.934，但公开 point
初始化输出面积仍为 0。因此该条件标为 protocol-invalid；其 24.37 J&F 不能解释
为有效 segmentation，也不能用于支持 gap 缩小。

## Correction / damage

K4 Dynamic expression 中：

- Temporal initial-wrong 后续纠正 35/52（67.31%）；Static 为 36/54
  （66.67%），差仅 +0.64pp。
- Temporal initial-correct 后被改坏 27/45（60.00%）；Static 为 25/43
  （58.14%）。
- 被 Temporal 纠正的 35 条 expression 上，Temporal−Static J&F 平均
  -1.08pp；离散 selection correction 没有形成像素收益。
- K4 Temporal 与 Static 只有 21/97（21.65%）Dynamic expression 的 candidate
  sequence 不同；大量 stage representation 变化没有越过当前 IoU scorer 的决策边界。

`per_stage_selection.csv` 保留每个 stage 的选择、初始纠正/破坏与 update 前后
anchor J&F。`qualitative_cases/` 按固定 identity 顺序展示 3 个 referent correction、
3 个 no-help、3 个 identity-damaged case；没有只挑成功案例。视觉抽查确认图中 RGB、
GT、Static-update K4 与 Temporal-update K4 对齐。

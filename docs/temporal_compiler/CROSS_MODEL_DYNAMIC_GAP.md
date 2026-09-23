# 跨模型 Dynamic Gap：64-object protocol check

## 当前结论

64 个预测无关抽样对象上的三模型 protocol check 已完整结束。三套官方模型均覆盖完全相同的 64 个对象、275 条官方表达，失败为 0。InstructSeg 与 VIRST 的 Dynamic−Static J&F 分别为 **−5.51pp** 和 **−4.02pp**，source-video cluster bootstrap 95% 区间均低于零；Sa2VA 在同一子集上为 **−1.40pp**，方向相同但区间跨零。

因此，pilot 结果暂时排除“明显只发生在 Sa2VA”的 Case C，更接近 **provisional Case A：Dynamic gap observed across multiple VLM families**。显式 temporal 的 VIRST 在绝对 J&F 上最高，但仍有显著 gap，pilot 不支持 Case B 所要求的“显式 temporal 模型已大幅缩小 gap”。这是 protocol check 的阶段判断，最终 Case 必须由正在运行的全部 274 paired objects 结果确认。

## 完整性与公平性审计

- 真实结果目录：`/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260922_cross_model_dynamic_gap_pilot64_3model`
- 子集：固定 seed=42 后按源视频抽样的 64 个 paired objects，覆盖 64 个源视频；样本选择不读取模型预测。
- 期望表达身份：275。Sa2VA、InstructSeg、VIRST 均为 275/275 成功、失败 0、缺失 0。
- 每模型完整 Static/Dynamic paired objects：64/64；三模型 shared complete set：64/64；集合逐 key 完全一致。
- 对每个对象，同一描述类型的多条表达先平均；对象等权。bootstrap 以 source video 为 cluster，固定 seed=42，2,000 次。
- 所有模型使用同一 official Static/Dynamic expression、target object 和 evaluation frames。GT mask 只进入独立评价路径。
- 跨模型绝对 J&F 不是因果比较：三模型保留各自官方/原生推理结构，主统计是模型内部 Dynamic−Static。

Sa2VA 原结果文件还包含 manifest 中其余 210 个对象，因此审计中的 `records_after_condition_filter=1189` 和 `extra_expression_identities` 是预期的源文件超集；汇总前严格按 pilot 的 275 个 identity 取交集，没有把额外对象计入本表。

## 主要结果

| 模型 | Static J | Static F | Static J&F | Dynamic J | Dynamic F | Dynamic J&F | Dynamic−Static J&F | 95% CI | Dynamic-worse 对象比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sa2VA-4B | 0.4709 | 0.4538 | 0.4624 | 0.4524 | 0.4444 | 0.4484 | −0.0140 | [−0.0382, 0.0078] | 0.5938 |
| InstructSeg | 0.5104 | 0.5057 | 0.5081 | 0.4505 | 0.4554 | 0.4529 | −0.0551 | [−0.0896, −0.0237] | 0.6719 |
| VIRST | 0.6705 | 0.6685 | 0.6695 | 0.6269 | 0.6318 | 0.6293 | −0.0402 | [−0.0755, −0.0090] | 0.5781 |

VIRST 的 absolute Dynamic J&F 比另外两模型高，不能据此声称它消除了描述类型差异；模型内部 gap 仍为 −4.02pp。反过来，模型训练数据、视觉编码器、输入帧协议和像素执行器都不同，因此 gap 大小也不能当作架构单因素消融。

## 官方模型与输入协议

| 模型 | 官方 checkpoint / code | 实际视频输入 | 分辨率 | 显式 temporal module | N=8/16/32 scaling |
|---|---|---|---|---:|---|
| Sa2VA-4B | 原始 Sa2VA-4B；checkpoint index SHA256 `6b8e6a52…` | 精确复用 manifest 的 N=16 VLM 帧；SAM2 输入、提示和传播固定 | VLM 每帧 401408–1605632 pixels；SAM2 1024 square | 否 | 复用上一轮公平 ablation；Dynamic N32−N8 = +0.392pp，95% CI [−0.40,+1.17]pp |
| InstructSeg | `weic22/InstructSeg` revision `d5375970…`；code `01343144…` | 每个输出帧使用当前帧和四个有序邻近 reference frames | SigLIP / Mask2Former 384，输出还原原视频大小 | 是 | N/A：官方 native per-target 5-frame protocol 不提供只改全局 VLM 帧预算的等价接口 |
| VIRST | 官方 checkpoint SHA256 `bd4a708…`；code `00aecefd…` | 最多 64 个 locked VLM frames、32 个 locked SAM prompt frames、全视频传播 | VLM anyres_nopad；segmentation 1024 | 是（STF + TDAU） | N/A：官方 native flex/propagation protocol 无法在不改核心算法时只改 VLM 帧预算 |

InstructSeg 和 VIRST 没有为完成 ablation 而修改核心架构。它们的官方路径未暴露可跨模型对齐的逐表达 visual-token 计数，也未在此次四卡并发 protocol check 中记录独立的同步 latency/peak-memory，因此这些字段记为 N/A，不据此给效率结论。

## 阶段 Case 判定

- **Case A（provisional）：支持。** 至少 InstructSeg 与 VIRST 两个不同官方家族存在区间低于零的 Dynamic gap；Sa2VA 点估计方向一致。
- **Case B：当前不支持。** 显式 temporal VIRST 没有在这 64 个对象上消除 gap。
- **Case C：当前不支持。** gap 不是仅在 Sa2VA 出现。
- **Case D：仍作为全量审计的保守备选。** 若 274-object 结果改变方向、出现大量失败或 shared paired set 不完整，将回退为 mixed / insufficient。

当前已启动全部 274 paired objects、1,189 expressions 的官方 InstructSeg 和 VIRST 四卡分片；完成后本报告将保留本节作为 protocol check，并另加全量主结果。最终 Method Route 在 SAM3.1 全量 coverage 同时完成前不选择。

## 随附轻量结果

- `results/cross_model_pilot64_summary.csv`
- `results/cross_model_pilot64_shared_summary.csv`
- `results/cross_model_pilot64_audit.json`
- `results/cross_model_pilot64_pair_overlap.json`
- `results/cross_model_pilot64_object_pairs.csv`
- `figures/cross_model_pilot64_gap.png`

逐帧 masks、视频、模型权重和大中间结果仅保存在 artifact 根目录，不进入 git。

# 跨模型 Dynamic Gap

## 全量主结果

全部 274 个 prediction-independent paired objects（90 个源视频、1,189 条官方表达）已在三套官方模型上完成。每套模型均为 1,189/1,189 成功、失败 0；三模型完整配对对象集合逐 key 相同。对象内同类型多表达先平均，对象等权，95% 区间由 source-video cluster bootstrap（seed=42，2,000 次）得到。

| 模型 | Static J | Static F | Static J&F | Dynamic J | Dynamic F | Dynamic J&F | Dynamic−Static J&F | 95% CI | Dynamic-worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sa2VA-4B | 0.4308 | 0.4301 | 0.4304 | 0.3961 | 0.3974 | 0.3967 | **−3.37pp** | [−4.80, −1.90]pp | 61.68% |
| InstructSeg | 0.4482 | 0.4471 | 0.4477 | 0.3976 | 0.4031 | 0.4004 | **−4.73pp** | [−6.38, −2.92]pp | 69.34% |
| VIRST | 0.5924 | 0.5950 | 0.5937 | 0.5875 | 0.5925 | 0.5900 | **−0.37pp** | [−2.68, +1.83]pp | 52.55% |

主判定为 **Case B（有限定）**：Sa2VA 与统一分割 VLM InstructSeg 都保留显著 Dynamic gap，而显式强化时序组织的 VIRST 将 gap 缩小到 −0.37pp，区间跨零。该结果不支持“主要是 Sa2VA 特例”，也不支持“所有模型家族都有同等 gap”。限定是：InstructSeg 自身也含局部 temporal 模块，因此这里支持的是 VIRST 的强时序协议/模块整体显著缩小 gap，不能把因果归给单个模块。

VIRST 的绝对 J&F 更高，但跨模型 checkpoint、训练数据、视觉编码器、输入协议和 pixel executor 均不同，绝对分数不是架构单因素消融。核心比较始终是各模型内部的 Dynamic−Static。

## 协议审计

- 真实汇总目录：`/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260922_cross_model_dynamic_gap_all274`
- 固定上一轮 Long-RVOS manifest；同一 video、object、official expression 和 evaluation frames；不依据预测筛样本。
- Sa2VA 精确复用 manifest N=16 VLM 帧和上一轮结果。
- InstructSeg 使用官方 native current-frame + 4 个有序邻近 reference frames。
- VIRST 使用官方 native flex 协议：实际 64 个 locked VLM frames、32 个 locked SAM2 prompt frames，再全视频传播。1,189 条记录的运行时审计均通过；同视频采样签名不一致数为 0，GT 进入模型记录数为 0。
- 三模型 paired objects 均为 274/274，缺失/失败均为 0，因此没有因不完整表达而改变配对统计。
- N=8/16/32 仅 Sa2VA 有公平接口：Dynamic N32−N8 为 +0.392pp，95% CI [−0.40,+1.17]pp。InstructSeg/VIRST 若硬改帧数会改变原生算法，故记 N/A。

## 模型来源与输入

| 模型 | 官方 checkpoint / code | 实际输入 | 显式 temporal 机制 |
|---|---|---|---|
| Sa2VA-4B | 原始 Sa2VA-4B；index SHA256 `6b8e6a52…` | manifest N=16；SAM2 输入/传播固定 | 无旧 temporal gate/head |
| InstructSeg | `weic22/InstructSeg` revision `d5375970…`；code `01343144…` | 每输出帧 5 帧，SigLIP/Mask2Former 384 | 有局部 temporal aggregation |
| VIRST | 官方 checkpoint SHA256 `bd4a708…`；code `00aecefd…` | 最多 64 VLM + 32 SAM2 prompt，全视频传播 | STF + TDAU |

逐模型 visual-token 数未由官方 InstructSeg/VIRST 路径暴露；此前并行运行没有可比的独立同步 latency/peak-memory，因此不据此作效率定量结论。VIRST 的输入规模与约 53GB 运行显存表明它明显重于本轮紧凑 Sa2VA 表示路径，但不能转换成正式吞吐倍率。

## 64-object protocol check（保留）

启动全量前的 64-object check 覆盖 64 个视频、275 条表达，三模型均 275/275 成功。Sa2VA、InstructSeg、VIRST 的 gap 分别为 −1.40pp（CI [−3.82,+0.78]）、−5.51pp（[−8.96,−2.37]）和 −4.02pp（[−7.55,−0.90]）。VIRST pilot 与全量判定不同，说明必须保留预注册全量扩展，不能按 pilot 选择论文叙事。

## 轻量结果

- `results/cross_model_all274_summary.csv`
- `results/cross_model_all274_shared_summary.csv`
- `results/cross_model_all274_audit.json`
- `results/cross_model_all274_pair_overlap.json`
- `results/cross_model_all274_object_pairs.csv`
- `figures/cross_model_all274_gap.png`

原始 masks、视频、权重和大中间结果不进入 git。

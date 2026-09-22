# Sa2VA 表示诊断

## 结论

在上一轮完全相同的 1,189 条官方表达和 N=8/16/32 帧索引上，Sa2VA 的分割 token 表示会随可见帧增加而移动，但这种移动没有稳定转化为 J&F 改善。当前证据不支持“`z_seg` 基本不变”的纯 representation bottleneck；更支持 **grounding-interface bottleneck**，同时不能排除模型没有把新增帧组织成可用于动态指代的时序状态。

这只是当前 Sa2VA、当前 Long-RVOS 诊断子集下的机制证据，不等价于“模型完全不理解时序”。最终 Method Route 仍须等待跨模型和 SAM3.1 candidate coverage 结果。

## 数据与审计

- 真实结果目录：`/9950backfile/chenjiahui/evo_artifacts/results/temporal_compiler/20260922_120123_sa2va_representation`
- 复用上一轮 manifest：90 个源视频、274 个对象、1,189 条官方表达，每条均有 N=8/16/32。
- 表示记录：3,567/3,567 成功，失败 0，唯一键 3,567，重复 0。
- 表达构成：Static 425、Dynamic 418、Hybrid 346。
- 每条主分析记录均满足 `seg_token_count == 1`、`segmentation_invoked == false`、`ground_truth_loaded == false`。
- 运行时代码未发现 temporal gate/head；模型加载无 missing、unexpected 或 mismatched key。
- extraction 运行代码 commit：`b35eaf647d98d1e549fbb098239c2d3cf12491d1`；checkpoint index SHA256：`6b8e6a52494ba5db4069693c31688f1895e02db36deca6de4c58c1143322ff87`。
- `normalized L2` 定义为两个向量各自单位归一化后的 L2 距离。bootstrap 固定 seed=42，以源视频为 cluster，共 2,000 次。

## 1. 同一表达随帧预算的表示变化

表内为 N=8 对 N=32 的表达级均值。

| 类型 | 空间 | cosine similarity | angular distance | normalized L2 | n |
|---|---:|---:|---:|---:|---:|
| Static | `h_seg` | 0.9572 | 0.2710 | 0.2696 | 425 |
| Static | `z_seg` | 0.9809 | 0.1695 | 0.1690 | 425 |
| Dynamic | `h_seg` | 0.9611 | 0.2630 | 0.2618 | 418 |
| Dynamic | `z_seg` | 0.9805 | 0.1752 | 0.1747 | 418 |
| Hybrid | `h_seg` | 0.9582 | 0.2698 | 0.2684 | 346 |
| Hybrid | `z_seg` | 0.9806 | 0.1720 | 0.1715 | 346 |

`text_hidden_fcs` 后的变化小于最后层 `[SEG]` hidden state，但不是零。以 Dynamic 为例，normalized L2 从 `h_seg` 的 0.2618 降到 `z_seg` 的 0.1747。

## 2. 变化尺度校准

| perturbation | 空间 | cosine similarity | angular distance | normalized L2 | 单位/n |
|---|---:|---:|---:|---:|---:|
| 同表达 N8 vs N32，Dynamic | `h_seg` | 0.9611 | 0.2630 | 0.2618 | expression/418 |
| 同表达 N8 vs N32，Dynamic | `z_seg` | 0.9805 | 0.1752 | 0.1747 | expression/418 |
| 同对象 Static vs Dynamic @N16 | `h_seg` | 0.9195 | 0.3657 | 0.3619 | object/274 |
| 同对象 Static vs Dynamic @N16 | `z_seg` | 0.9528 | 0.2575 | 0.2555 | object/274 |
| 同视频不同对象 @N16 | `h_seg` | 0.6674 | 0.8182 | 0.7914 | source video/87 |
| 同视频不同对象 @N16 | `z_seg` | 0.7370 | 0.7157 | 0.6968 | source video/87 |

Dynamic 的 N8→N32 `z_seg` normalized L2 是同对象 Static↔Dynamic 变化的约 68%，也是同视频不同对象变化的约 25%。因此，新增四倍可见帧确实改变了送往分割接口的向量，但变化远小于改变目标对象带来的移动。

## 3. 表示变化与分割收益

下表为 N8→N32 normalized L2 与 `DeltaJF = JF32 - JF8` 的 Spearman 相关；区间按源视频 cluster bootstrap。

| 类型 | 空间 | rho | 95% CI | n/视频 |
|---|---:|---:|---:|---:|
| Static | `h_seg` | 0.0402 | [-0.0896, 0.1588] | 425/90 |
| Static | `z_seg` | 0.0495 | [-0.0702, 0.1598] | 425/90 |
| Dynamic | `h_seg` | 0.1194 | [-0.0059, 0.2333] | 418/90 |
| Dynamic | `z_seg` | 0.0902 | [-0.0261, 0.1988] | 418/90 |
| Hybrid | `h_seg` | -0.0001 | [-0.1041, 0.1196] | 346/87 |
| Hybrid | `z_seg` | 0.0058 | [-0.1038, 0.1231] | 346/87 |

所有区间均跨零。Dynamic 的点估计略为正，但现有样本不足以支持“表示移动越大，J&F 改善越多”。

## 4. N8 与 N32 掩码变化

| 类型 | Prediction IoU | 像素 disagreement | 绝对面积变化 | 表达级 DeltaJF 均值 |
|---|---:|---:|---:|---:|
| Static | 0.7808 | 0.0206 | 0.0144 | -0.0072 |
| Dynamic | 0.7940 | 0.0188 | 0.0135 | +0.0045 |
| Hybrid | 0.7932 | 0.0170 | 0.0122 | -0.0029 |

Dynamic 的 `z_seg` 有可测移动，但预测掩码仍较稳定（Prediction IoU 0.7940、逐帧像素分歧 1.88%），而且表示距离与 DeltaJF 无可靠相关。这一组合最符合“表示变化没有被现有单 token→像素执行接口稳定利用”。“temporal organization bottleneck”仍是可能的上游解释，但仅靠单个汇总向量的距离不能区分它与更纯粹的 grounding interface 问题。

## 判定

- **纯 representation bottleneck：不支持。** `z_seg` 随 N8→N32 的移动并非近零。
- **grounding-interface bottleneck：当前最受支持。** `z_seg` 移动没有稳定对应 J&F 或明显 mask 改变。
- **temporal organization bottleneck：尚不能排除。** 距离只能说明状态变化，不能证明变化是否编码了事件次序、阶段或跨帧关系。
- **下一 gate：** 需要 InstructSeg/VIRST 的同对象 Static–Dynamic gap 与 SAM3.1 candidate coverage，之后才能在 Route A/D/E 间选择。

## 随附文件

- `results/representation_summary.csv`
- `results/representation_calibration_summary.csv`
- `results/representation_jf_correlations.csv`
- `results/prediction_mask_change_summary.csv`
- `results/representation_analysis_audit.json`
- `figures/representation_scale_calibration.png`
- `figures/representation_vs_jf.png`
- `figures/mask_stability_vs_jf.png`

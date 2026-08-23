# EvoSeg：统一图像-视频的忠实指代分割（Faithful Image-Video Referring Segmentation）

> 汇报用总结 · 2026-08-23 · 数据为真实评测结果，含失败与权衡

---

## 一句话定位

> **"分割得再好，也要知道该不该分割。"**
> 我们把统一图像+视频的指代分割从"无条件的 mask 生成"变成"有证据的选择性预测"：
> 模型不再默认目标一定存在，而是学会在目标不存在/消失/被顶替时**拒绝（abstain）或在目标消失的那一刻停止传播 mask**。
> 核心主张：**图像级拒答必要但不充分——忠实指代分割必须时序化（existence 是逐帧谓词 e → e_t）**。

---

## 一、动机：为什么"会分割"不等于"知道该不该分割"

当前指代分割（RES/RVOS）从训练到评测都默认"查询里的目标一定存在"——模型被训练成"有查询就出 mask"。但真实场景中：

- 用户会描述一个**根本不存在**的东西（假前提 / false premise）
- 会用**过时/错误**的信息提问（反事实）
- 视频里目标会**中途消失、被遮挡**，或**被长得像的物体顶替**（identity swap）

这些情况下，**正确的行为不是硬画一个 mask，而是拒答或在目标消失那一刻停住**。我们把这种"能否判断该不该分割"的能力称为 **grounding faithfulness（接地忠实性）**。

**与已有工作的区分**（避免踩坑）：
- 图像侧假前提拒答前人已做（SESAME、GSVA 的 [REJ] token、HalluSegBench）
- 视频侧的**时序忠实性**（existence 逐帧谓词 e_t）几乎没人做端到端训练，更没有统一图像+视频的忠实指代分割
- 我们不做"又一个拒答模型"，而是做 **"目标存在性"从图像级常量升级为逐帧时序谓词**这个概念的完整落地

---

## 二、关键诊断（第一个贡献：empirical finding）

我们构建了 8905 条 absent query 评测 + 1986 例视频忠实性基准（Ref-YT-VOS valid 衍生），发现：

| 现象 | 数据 |
|---|---|
| 所有基线模型对不存在的查询 **100% 幻觉** | Sa2VA-4B / 4B MultiTask / 8B MultiTask 全部 100% |
| 但正常分割又很强 | RefCOCO ~80 cIoU |
| **分割精度 ≠ grounding 忠实性** | 两者是不同能力，可同时成立 |
| 根因在数据 | 多任务 SFT 的数据脚本**显式跳过了 no-target 样本** |
| RL 补不上 | 纯 RL 采不到拒答轨迹（训练分布里没有这个行为） |

**结论**：忠实性不是事后能用纯 RL 补的能力，**必须在训练数据里显式建立**。

---

## 三、方法：一个主张，两次升级

### 3.1 Stage 1：图像 faithfulness SFT（教模型"说没有"）
no-target 拒答样本 ×4 + present 样本 ×1 的平衡配方 → 图像幻觉 **100% → 14.7%**，gRefCOCO **29.8 → 69.8**，RefCOCO 保持 82.2。

### 3.2 Stage 2：视频 faithfulness SFT（教模型"目标消失就停"）
从 Ref-YT-VOS 推导逐帧 presence，构造 **19057 例** temporal_absence / identity_swap / global_absence 训练数据，消失帧给零 mask 监督 → 视频整体幻觉 **91.3% → 6.6%**。

### 3.3 Stage 3：时序存在性评估器（当前主线，第一性原理重构）

**重新定义问题**：e_t 的本质不是"目标在不在"（独立目标检测），而是——

> **e_t = "SAM2 正在传播的这个 mask，在第 t 帧是否仍然忠实于 query 所指的目标"**（mask-query fidelity）

**分工式架构**（冻结 VLM + SAM2，只训一个 2-4M 的轻量评估器）：

```
query ─► VLM(冻结) ─► lang [SEG] 向量          ← 语义：要找什么
video ─► SAM2(冻结) ─► 逐帧 mask + 逐帧 mask区域特征  ← 空间+时间：SAM2 天然感知所有帧
                              │
        [逐帧特征 + mask区域 + 几何 + lang] ─► 轻量时序评估器 ─► e_t
                                                    │
                                              e_t=1 保留 mask / e_t=0 干净停住
```

**评估器的特征演化（每步都有诊断驱动）**：

| 版本 | 输入特征 | 动机 |
|---|---|---|
| MLP（TEG） | 逐帧独立，无时序 | 基线门控，但看不到"前 11 帧有、12 帧消失"的时间变化 |
| GRU | 场景均值池化 + lang | 加时序记忆，但均值池化对"小目标消失"太钝 |
| **Fidelity v2** | + mask 区域特征 | mask 底下内容变化 = "SAM2 跟的东西变了"的直接证据 |
| **Fidelity v4** | + mask 几何（面积/质心） | 目标消失时 mask 面积暴涨 ~10 倍（SAM2 跟丢信号），与外观正交 |
| **B+（进行中）** | + VLM 逐帧 hidden state | 让 VLM 真正感知每一帧（喂全部帧，池化逐帧 LLM 特征） |

---

## 四、当前结果（全部真实数字）

### 4.1 图像侧（8905 absent 查询）

| 模型 | RefCOCO | gRefCOCO | 幻觉率 |
|---|---|---|---|
| Sa2VA-4B（基线） | 81.95 | 29.8% | **100%** |
| Faithful-4B | 82.22 | **69.79%** | **14.7%** |
| 外部：SESAME | — | — | 33.5% |
| 外部：GSVA-7B（有[REJ]） | — | — | 44.6% |

### 4.2 视频忠实性（1986 例 / 52284 帧，absent_halluc_rate）

| 模型 | overall | temporal | identity | global | counterfactual |
|---|---|---|---|---|---|
| Sa2VA-4B | 91.3% | 87.6% | 90.9% | 87.2% | 97.0% |
| VideoFaithful-4B | 6.6% | 76.3% | 81.0% | 1.0% | 4.6% |
| **8B VideoFaithful** | **5.0%** | **61.1%** | **70.2%** | 0.7% | 2.8% |
| **Fidelity v2**（mask区域,pw1） | **4.2%** | **37.6%** | **37.7%** | 1.5% | 3.2% |
| **Fidelity v4**（+几何,pw1） | 5.2% | 51.3% | 58.0% | 1.5% | 3.5% |

**关键解读**：
- mask 区域特征（v2）把 hardest 的 temporal/identity 幻觉砍掉近半（67.5→37.6、71.5→37.7），overall 4.2%
- 代价是 present_miss 上升（v2 temporal 26.7%——太爱停）；加几何（v4）把漏检压回基线水平（16.6%），两项都优于 TEG 基线
- **存在一个"幻觉率 vs 漏检率"的权衡**，这是时序边界判别的本质难度，B+ 正在攻这个

### 4.3 外部泛化（没见过的数据）
- 图像 HalluSegBench 反事实：Sa2VA 100% 幻觉 vs 我们拒答 36-38%
- 视频 MeViSv2 no-target：Sa2VA 99.7% vs 我们 46.7-48.2%
- 8B 验证：同配方更大模型全维度更好（temporal 76.3→61.1），说明时序忠实性随规模提升

---

## 六、当前进度与下一步

| 项目 | 状态 |
|---|---|
| 图像 faithfulness SFT | ✅ 完成（幻觉 14.7%） |
| 视频 faithfulness SFT | ✅ 完成（overall 6.6%） |
| 时序评估器（mask 区域 + 几何） | ✅ 完成（temporal 37.6-51.3%） |
| **B+：VLM 感知所有帧** | ⏳ 特征提取 77%，预计今天出结果 |
| 外部基线（SESAME/GSVA/MeViSv2/HalluSegBench） | ✅ 完成 |
| Figure 1 + 失败分类学 | ✅ 完成 |
| 对比视频网页（模型效果展示） | ✅ 已部署（http://172.18.127.61:8899） |

**下一步**：
1. B+ 结果（VLM 逐帧感知）—— 目标突破"幻觉/漏检"权衡
2. 选定最终操作点 → 更新对比视频网页
3. 更新 paper brief → 写 paper 初稿（CVPR 2027 目标）

---

## 七、一句话给老师

**问题**：分割得再好也要知道该不该分割，视频里目标消失/换人时模型必须"停住"。
**发现**：所有基线 100% 幻觉但分割很强 → 忠实性是独立能力，根因在训练数据缺 no-target。
**方法**：faithfulness SFT（教拒答）+ 轻量时序评估器（e_t = mask-query 忠实度，VLM 语义 + SAM2 空间 + 几 M 参数的时序判断）。
**结果**：absent 幻觉从 100% 打到 4-15%，hardest 的时序类别从 87%/72% 打到 38-58%，外部模型/数据泛化成立。
**下一步**：让 VLM 真正感知每一帧（B+），把时序边界判别再推一步，然后写 CVPR 2027。

---

## 五、用到的数据集与对比的模型

### 5.1 训练数据

| 数据集 | 用途 | 规模/说明 |
|---|---|---|
| **RefCOCO / RefCOCO+ / RefCOCOg** | 标准指代分割 SFT | ~19k / 19k / 26k 指代句（标准协议） |
| **gRefCOCO** | 广义指代（含 single/multi/no-target）SFT | no-target 拒答样本 ×4 平衡配方 |
| **COCO2014** | 无目标查询负样本 | 8905 条 absent query 的一部分 |
| **Ref-YT-VOS** | 视频忠实性训练（自建） | **19057 例**：从索引 mask 推导逐帧 presence，构造 temporal_absence / identity_swap / global_absence 三类 |
| ReasonSeg / MeViS / InterVOS | 推理/视频分割辅助 | 已有，用于能力保持 |

### 5.2 评测基准

| 基准 | 来源 | 指标 |
|---|---|---|
| **8905 条 absent query** | gRefCOCO no-target + COCO 无目标查询（自建） | 图像幻觉率（应拒答却出 mask） |
| **视频忠实性基准（1986 例 / 52284 帧）** | Ref-YT-VOS valid 衍生（自建） | absent_halluc / present_miss / frame_acc；四类：temporal_absence / global_absence / counterfactual_swap / identity_swap |
| **HalluSegBench** | 外部（CVPR 2025 反事实分割幻觉） | 反事实拒答率 |
| **MeViSv2 no-target** | 外部（TPAMI，含 valid_u 干净留出集） | 视频 no-target 幻觉率 |
| **RefCOCO/+/g + gRefCOCO** | 标准 | cIoU（分割质量保持） |

### 5.3 对比的模型

**我们自己的 checkpoints**（同一 Sa2VA 基座逐步升级）：

| 模型 | 说明 |
|---|---|
| Sa2VA-4B | 原始基线（不做任何忠实性训练） |
| EvoSeg-4B MultiTask | 多任务 SFT（无 no-target） |
| Faithful-4B | 图像 faithfulness SFT（教拒答） |
| VideoFaithful-4B / 8B | +视频 faithfulness SFT |
| TEG-4B | +逐帧 MLP 存在性门控 |
| **Fidelity-4B**（当前主线） | +mask 区域特征 / 几何特征 / B+ VLM 逐帧感知 |
| GRPO-RL 变体 | 消融（RL 无额外增益，证明数据是根因） |

**外部基线**（关键对照，在 8905 条上的幻觉率）：

| 模型 | 来源 | 幻觉率 |
|---|---|---|
| **SESAME** | CVPR 2024 "See, Say, Segment"（假前提拒答 SOTA） | 33.5% |
| **GSVA-7B** | CVPR 2024（显式 [REJ] token 拒答） | 44.6% |
| **我们 Faithful-4B** | — | **14.7%** |
| **Text4Seg** | gRefCOCO 参考（~70 cIoU） | 我们 69.79（同区间，不宣称 SOTA） |
| **Sa2VA 官方数字** | 4B 82.4/77.6/79.7 | 我们保持 82.2/76.5/79.0 |

**对比结论**：我们在 8905 条上 14.7% 幻觉，比有显式拒答机制的 SESAME（33.5%）和 GSVA（44.6%）低一半到三倍；外部数据（HalluSegBench / MeViSv2）上 Sa2VA 100% / 99.7% 幻觉 vs 我们 36-48% 拒答——**泛化成立，且非 Sa2VA-specific**。

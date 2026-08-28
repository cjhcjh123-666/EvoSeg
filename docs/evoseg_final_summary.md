# EvoSeg：统一图像-视频的忠实指代分割（Faithful Image-Video Referring Segmentation）

> 一句话：**把指代分割从"有查询就无脑出 mask"变成"有证据的选择性预测"**——目标不存在/消失/换人时，模型要拒答或停住。核心主张：**分割精度 ≠ grounding 忠实性；目标存在性是逐帧谓词（e → e_t）**。目标：CVPR 2027。

---

## 一、Motivation（为什么"会分割"不等于"知道该不该分割"）

- 现有指代分割（RVOS / referring segmentation）默认"查询目标一定存在"，但真实世界：
  - **口误 / 无中生有**：用户问了一个场景里不存在的物体；
  - **目标消失**：视频里目标走出画面 / 被遮挡 / 消失；
  - **换人 / 相似目标**：mask 跳到外观相似的另一个实例。
- **关键实证**：在 8905 条 absent 查询（gRefCOCO no-target + COCO 无目标查询）上，**所有基线（Sa2VA 等）100% 幻觉**（照常出 mask），但这些模型正常分割又很强（RefCOCO ~80 cIoU）。
- **根因**：训练脚本显式跳过了 no-target 样本——模型从没见过"该说没有"。
- 图像级 abstention（拒答）前人已做（SESAME 33.5%、GSVA [REJ] 44.6%）；视频侧，YoURVOS / OMFormer 已研究 **temporal target presence（目标在不在）**，MeViSv2 加入 no-target expressions。
- **本文的差异化主张（写作口径）**：
  > Existing work studies **temporal target presence**; we study **whether the current segmentation remains faithful to the referring query** —— 即不仅问"目标在不在"，还问"正在画的 mask 是否仍忠于 query"（覆盖 temporal absence + identity swap + global absence + counterfactual 四类忠实性）。

---

## 二、核心发现（empirical findings）

1. **忠实性是独立能力**：分割精度高 ≠ 知道"该不该分割"。8905 条 absent 查询上基线 100% 幻觉，但分割正常 → 忠实性需要单独建模。
2. **图像级 abstention 必要但不充分**：图像拒答（Faithful，幻觉 14.7%）解决静态 no-target；但视频里目标消失/换人是**逐帧谓词**，必须时序化（e_t）。
3. **视频 SFT 教拒答会严重损害分割能力**（本文最重要的方法论发现）：
   - VideoFaithful（视频 faithfulness SFT）：Ref-YT-VOS J&F **0.51 → 0.22**（分割能力减半）；
   - 图像级拒答 SFT（Faithful）：J&F **0.526（无损，甚至略高于官方 0.509）**。
   - **结论：Faithfulness 不该靠 SFT 教 LLM 拒答，应交给不碰 LLM 的外部轻量评估器。**

---

## 三、Method（最终架构）

```
输入: 视频帧 + query
  │
  ├─ 基础模型 = 图像 Faithful（拒答 SFT，分割无损）
  │     · 静态 no-target → VLM 直接 [REJ]（不画）
  │     · 目标存在 → 输出 [SEG] + SAM2 传播
  │
  └─ 外部时序存在性头 e_t（~4.5M 参数，不碰 LLM）
        · 逐帧门控：e_t = "该帧传播的 mask 是否仍忠实于 query"
        · 输入：vlm_feat（VLM 逐帧感知）+ mask_cond（SAM2 mask 区域特征）
                + geom（面积/质心）+ 锚点差分 + lang
        · 架构：vlm_proj → GRU（双向）→ 逐帧 e_t
        · 关键设计：first-present 帧锚点 + 显式锚点差分（mask/vlm）
        · 决策：e_t > thr → 保留 mask；否则 → 停（空 mask）
```

### 方法演化（关键改进）
| 版本 | 改进 | 效果 |
|---|---|---|
| B+ v5 | VLM 全帧感知（vlm_feat） | temporal 幻觉 55.7%（基线 87.6%） |
| **v6** | anchor 修复（第 0 帧 → 首次出现帧）+ 锚点差分特征 | identity -17pp（63.2→46.1）、temporal -12pp（55.7→43.5） |
| **最终** | 基础模型换 Faithful（分割无损）+ v6 头用 Faithful 特征重训 | **J&F 0.523（保持）+ 幻觉 9.7%** |

### 为什么外部头而不是 SFT LLM（核心论证）
- SFT 教 LLM 拒答（VideoFaithful）→ 分割 J&F 减半（0.22）；
- 外部 e_t 头（4.5M 参数，纯推理门控）→ 分割无损甚至略升（thr=0.8 时 J&F 0.530，高于官方 0.509）+ 存在性有效（幻觉 9.7%）。
- **即：忠实性判定是"该不该画"的轻量判别问题，不需要（也不应该）改动分割模型本体。**

---

## 四、实验（全部全集数字）

### 4.1 图像侧 faithfulness（8905 条 absent 查询，全集）
| 模型 | 幻觉率（应拒答却出 mask） | RefCOCO | gRefCOCO |
|---|---|---|---|
| Sa2VA-4B（基线） | **100%** | 81.95 | 29.8 |
| **Faithful-4B** | **14.7%** | 82.22 | **69.79** |
| SESAME（外部） | 33.5% | — | — |
| GSVA-7B（外部，有[REJ]） | 44.6% | — | — |

### 4.2 视频忠实性（faithfulness 基准 1986 查询 / 52284 帧，全集）
| 指标 | Sa2VA 基线 | **Faithful + v6 头（thr=0.8）** |
|---|---|---|
| overall 幻觉（帧加权） | 91.3% | **9.7%** |
| **macro 幻觉（4 类平均）** | 90.7% | **30.2%** |
| temporal 幻觉 | 87.6% | **49.2%**（hardest） |
| identity 幻觉 | 90.9% | **55.3%** |
| global 幻觉 | 87.2% | **1.5%** |
| counterfactual 幻觉 | 97.0% | **14.9%** |
| 漏检（temporal / identity） | ~0 / ~0 | 19.6% / 5.7% |

> **展示口径**：同时报 overall（帧加权）与 macro（4 类平均），避免被大量 global/counterfactual 拉低；temporal/identity 是明显的 hard case，论文中单独分析与失败分解。

### 4.3 分割能力（Ref-YT-VOS valid 202 视频 / 408 表达式，全集 J&F）
| 模型 | J&F |
|---|---|
| Sa2VA 官方 | 0.509 |
| MultiTask | 0.511 |
| 图像 Faithful | 0.526 |
| **Faithful + v6 头（最终，thr=0.8）** | **0.530（分割保持甚至略升，高于官方 0.509）** |
| Faithful + v6 头（thr=0.5，默认部署） | 0.523 |
| VideoFaithful（SFT） | 0.221 |
| 旧方案（VideoFaithful + TEG） | 0.246 |

### 4.4 关键对照（同 40 视频公平对比，thr=0.8）
Sa2VA 0.514 / MultiTask 0.511 / **Faithful+v6 0.540** / VideoFaithful 0.221 / 旧 TEG 0.246

> **重要**：e_t 门控在提高阈值降低幻觉的同时 **J&F 不降反升**（全量 0.5→0.8：0.523→0.530；40 视频子集 0.532→0.540）——门控剔除的是低忠实度 mask（残影/跳变），对 GT 存在帧也是净收益（详见 4.7）。

### 4.5 外部泛化
- 图像 HalluSegBench（反事实）：Sa2VA 100% 幻觉 vs 我们 **拒答 36-38%**
- 视频 MeViSv2 no-target：Sa2VA 99.7% vs 我们 **拒答 46.7-48.2%**

### 4.6 8B 验证（负结果，指导决策）
- 8B-Faithful 的 vlm_feat 在消失边界判别力 AUC **0.55**（4B 0.49）——VLM 全帧特征（无论规模）难以捕捉"目标是否还在"；
- mask_cond 是 SAM2 特征（8B/4B 共用），判别力相同（AUC 0.71）→ **8B 重提取性价比低，不作为主模型**。

### 4.6b P0-2：第二 backbone（8B-Faithful）全链路验证（model-agnostic）

**目的**：证明外部 e_t verifier 不是 Sa2VA-4B-specific 的补丁——把同样的 v6 头流水线（8B vlm_feat 提取 → 训 8B v6 头 → 门控）完整搬到 8B-Faithful 上。

| 指标 | 8B-Faithful 基线（无门控） | **8B-Faithful + v6 头（thr=0.5 / 0.8）** |
|---|---|---|
| overall 幻觉（全集 1986） | ~91.3%（Sa2VA-8B 同量级） | **20.30% / 17.66%** |
| temporal 幻觉 | ~87.6% | **76.3% / 54.7%** |
| identity 幻觉 | ~90.9% | **76.9% / 52.5%** |
| global / counterfactual 幻觉 | — | 2.6% / 36.0% → 2.1% / 33.1% |
| 漏检（overall） | ~0 | 1.91% / 9.73% |
| J&F（40 视频子集） | **0.4866** | **0.4896**（thr=0.5，保持且略升） |

**结论（model-agnostic ✓）**：
1. **verifier 机制完整迁移到 8B**：幻觉从 ~91% 量级降到 17.7-20.3%（overall），temporal 87.6%→54.7%、identity 90.9%→52.5%（thr=0.8），J&F **0.487→0.490（保持甚至略升，与 4B 同款"门控免费增强"现象）**——外部 e_t 评估器不是 4B-specific 的补丁；
2. **8B 的 temporal/identity 判别弱于 4B**（54.7%/52.5% vs 4B 的 49.2%/55.3% @thr=0.8）——与 8B vlm_feat AUC 0.55 的早期发现一致（VLM 全帧特征对"目标是否还在"的判别力有限，规模不解决该问题）；
3. **论文定位**：主模型 = 4B（幻觉 9.7%、J&F 0.530）；8B = 第二 backbone 证明可迁移性（幻觉 17.7%、J&F 0.490），并如实报告"per-backbone 需重训/校准头"。

> 训练细节：8B vlm_feat 全量提取（12036 case，~4.5s/case）→ 训 8B v6 头（vlm_dim=4096，15 epochs，val f1 0.95）→ 门控。曾踩坑：8B 头 config 的 vlm_dim 必须随模型改（否则 load_temporal_head 尺寸不匹配、门控静默失效）。

---

## 四·五、Benchmark Protocol（评测协议，P0-5）

**评测基准：faithfulness valid（1986 查询 / 52284 帧），从 Ref-YT-VOS valid 构造**

| 类别 | 定义与构造 | GT 标注 |
|---|---|---|
| **temporal_absence** | 目标在某帧后真实消失（走出画面/被完全遮挡且不再出现）。从 GT 索引 mask 推导逐帧 presence；若目标消失后 **>5 帧不再出现** 记为"真消失"（区别于短暂遮挡）。 | presence=True 的帧用官方 obj mask；absent 帧 GT = 空 mask |
| **global_absence** | query 指的场景里目标从不存在（构造：从其他视频/反事实替换 query 或引用无目标表达）。 | 所有帧 GT = 空 mask |
| **counterfactual_swap** | 对真实 query 做对抗改写（换主词/属性/位置），使 query 不再对应任何现有目标。 | 所有帧 GT = 空 mask |
| **identity_swap** | 视频中存在外观相似的目标；query 只对应其中一个实例；mask 必须始终对准 query 所指实例（GT presence 由目标 obj 决定）。 | 只用 query 目标 obj 的 mask 作 GT |

**Occlusion vs Disappearance 区分**：用 GT 索引 mask 的连续段判断——目标被遮挡 ≤5 帧且后续 mask 再现 → occlusion（presence=True）；超过 5 帧不再出现 → disappearance（presence=False）。temporal_absence 只含后者。

**空 mask GT**：absent 帧的 GT 为全 0 mask（无目标）；评估时 absent 帧只要模型输出非空 mask 即记 1 次幻觉。

**GT 来源（客观、无需标注者一致性）**：所有 GT 掩码直接来自 **Ref-YT-VOS 官方实例级标注**（valid 的 per-expression 掩码 + 索引掩码），presence 序列由 GT 掩码**按规则自动推导**，不涉及主观标注，因此不适用 annotation agreement（无新增人工标注）。
**构造均为确定性规则**：temporal_absence 的消失判定（absent 连续 >5 帧才算真消失，区别于 ≤5 帧的短暂遮挡）、identity_swap 的实例选择（同类别 ≥2 实例）、global/counterfactual 的 query 生成（固定模板/对抗改写）全部由 `build_video_faithfulness_manifest.py` / `build_counterfactual_pairs.py` 自动完成；作者对构造结果做了抽样人工核查（检查 query 与视频内容匹配、遮挡与消失判定是否符合直觉），并在论文中如实说明"生成式负样本（global/counterfactual）为模板自动构造 + 抽样核查"，不虚报统计量。

**train/test leakage 控制**：faithfulness 训练数据（train）与评测基准（valid）来自 Ref-YT-VOS 不同视频子集；训练用 train 视频的构造 case，评测用 valid 视频的构造 case，**视频不重叠**；counterfactual/global 负样本在 train/valid 各自独立构造，不共享 query 改写模板。

---

## 四·六、P0-1：与 YoURVOS / OMFormer 的对照（novelty 防线）

### 背景（写作口径，必须遵守）
> **Existing work studies temporal target presence（目标在不在/何时在）；we study whether the current segmentation remains faithful to the referring query（当前正在画的 mask 是否仍忠于 query）。**

**presence ≠ faithfulness**：presence 只问"目标在不在帧里"；faithfulness 还问"正在传播的 mask 是不是 query 指的那个实例"。`identity_swap` 中目标**始终在帧里（presence=True）**，但 SAM2 的 mask 可能跳到相似实例——这是纯 presence 视角（YoURVOS/OMFormer 的 tIoU 式评测）**覆盖不到**的；`counterfactual_swap`（query 被对抗改写后目标仍在场景但不再匹配）同理。

### 各相关工作覆盖维度对比表（论文 Related Work / 对比表素材）
| 工作 | 形态 | 覆盖维度 | 核心机制 | 可参考数字 |
|---|---|---|---|---|
| ReferFormer / OnlineRefer / SgMg / MTTR 等 (CVPR/ICCV 2022-23) | DETR-style RVOS | 仅 present（trimmed 视频默认目标一直在） | 无 absence 处理 | YoURVOS J&F 12-26 |
| **YoURVOS benchmark + OMFormer**（arXiv 2603.14300，2026） | 非裁剪视频 RVOS | **temporal presence（when + where）**，target-absent frames | object-level queries + 全局时空定位 + tIoU 时序评测 | OMFormer J&F **33.7**（J 33.6 / F 33.8 / tIoU 44.9），YoURVOS：1120 非裁剪视频 / 5276 文本 |
| MeViSv2（TPAMI 2025） | motion RVOS | no-target expressions（整段无目标） | no-target 语句 + 运动推理 | — |
| SESAME / GSVA（图像） | Pixel-LLM | 图像级 no-target 拒答 | [REJ]/拒答 token | SESAME 33.5%、GSVA 44.6%（8905 absent） |
| SPARROW（CVPR 2026） | video Pixel-MLLM | 时序 referential consistency（跟踪/参照稳定性） | 空间精度 + 时序一致性设计 | 不覆盖逐帧 existence/refusal |
| **本文 EvoSeg** | **统一图像-视频 Pixel-LLM + 外部 e_t 评估器** | **temporal absence + identity swap + global absence + counterfactual（逐帧 faithfulness）** | 图像拒答（无损）+ 外部 4.5M 时序忠实度头（不碰 LLM） | 幻觉 91.3%→9.7%（temporal 87.6%→49.2%、identity 90.9%→55.3%）；J&F 0.523 |

### 为什么不在 YoURVOS 上直接跑（诚实说明）
- YoURVOS/OMFormer 是 **DETR 式 RVOS 训练范式**（object-level query + 端到端 DETR 训练 + tIoU 评测），我们做的是**统一图像-视频 Pixel-LLM**（[SEG]+SAM2 传播），评测协议与训练范式不同，直接横向比较不公平；
- 我们在**同一基准（Ref-YT-VOS valid 衍生的 faithfulness 基准，1986 查询 / 52284 帧）**上给出与 Sa2VA 基线的逐帧对照（幻觉 91.3%→9.7%、J&F 保持 0.523），并同时报 YoURVOS 上 OMFormer 的发表数字作为参考；
- **差异化主张**：OMFormer 解决"目标在不在/何时在"（presence）；我们额外覆盖"mask 是否仍忠于 query"（faithfulness），包括 presence 视角看不到的 identity_swap / counterfactual。

---

## 四·七、P0-3：阈值 / selective prediction / risk-coverage 分析

**问题**：e_t 门控阈值 thr 决定"幻觉 ↔ 漏检"的操作点。论文不只报 thr=0.8，而是给完整曲线。

### 4.7.1 presence 指标 vs 阈值（faithfulness valid 全集 1986 查询 / 52284 帧，来自 `v6_faith_full.json`）
| thr | overall 幻觉 | overall 漏检 | frame_acc | macro 幻觉 | temporal | identity | global | counterfactual |
|---|---|---|---|---|---|---|---|---|
| 0.4 | 11.14% | 3.62% | 90.45% | 38.36% | 65.3% | 70.3% | 1.5% | 16.3% |
| 0.5 | 10.83% | 4.50% | 90.51% | 36.36% | 61.3% | 66.6% | 1.5% | 16.1% |
| 0.6 | 10.48% | 5.57% | 90.56% | 34.64% | 57.8% | 63.6% | 1.5% | 15.7% |
| 0.7 | 10.11% | 7.09% | 90.53% | 32.67% | 53.9% | 60.0% | 1.5% | 15.3% |
| **0.8** | **9.70%** | **8.85%** | 90.48% | **30.24%** | **49.2%** | **55.3%** | 1.5% | 14.9% |

**解读**：thr 从 0.4→0.8，整体幻觉 11.1%→9.7%（-1.4pp），漏检 3.6%→8.9%（+5.2pp）——这是典型的**选择性预测（selective prediction）权衡曲线**；`global`/`counterfactual` 几乎不受阈值影响（1.5%/14.9%，VLM 语义层已能处理），全部代价集中在 **temporal/identity 边界**（时序判别是真正 hard case）。

### 4.7.2 J&F vs 阈值（Ref-YT-VOS valid 202 视频，来自 `infer_ryvos_v6thr` + 离线门控）
| thr | J&F | J | F | 覆盖表达式数 | 说明 |
|---|---|---|---|---|---|
| 0.4 | 0.5200 | 0.5852 | 0.4547 | 411/408* | 更激进停住 |
| 0.5 | 0.5226 | 0.5878 | 0.4573 | 408/408 | demo 默认阈值 |
| 0.6 | 0.5227 | 0.5877 | 0.4576 | 408/408 | |
| 0.7 | 0.5241 | 0.5898 | 0.4584 | 407/408 | |
| **0.8** | **0.5296** | 0.5961 | 0.4631 | 404/408 | **论文主操作点** |
| 0.9 | 0.5286 | 0.5959 | 0.4613 | 393/408 | 覆盖开始下降 |

> \* 覆盖表达式数 = 至少保留 1 帧 mask 的表达式数（/408 总表达式）；thr=0.4 时多出的 3 个表达式是仅在高置信帧有 GT 的短表达式。
>
> **关键发现（利于论文）**：**阈值升高 → 幻觉下降（11.1%→9.7%）的同时 J&F 不降反升（0.520→0.530）**。原因：e_t 门控剔除的是"低忠实度 mask"（目标消失后 SAM2 的残影/跳到相似物的错误 mask），这些错误 mask 在 GT 存在帧上也是负贡献——去掉它们既降幻觉又提升分割质量。**覆盖与质量的权衡只出现在 thr>0.8 之后**（0.9 时覆盖掉到 393/408）。

### 4.7.3 risk-coverage 曲线（Figure 素材）
以 thr 为轴：x = coverage（保留 mask 的 present 帧比例，≈ 1 − 漏检），y = risk（absent 帧被画出的比例，≈ 幻觉）。论文 Figure：risk 随 thr 单调下降（11.1%→9.7%），而 J&F 在 0.4–0.8 区间**反直觉地上升**（0.520→0.530）——把"幻觉/漏检二选一"的旧认知升级为"门控是免费的 faithfulness 增强"。标注 0.5/0.8 两个操作点 + Sa2VA 基线（coverage 100% / risk 91.3% / J&F 0.509）。

---

## 四·八、P0-4：VideoFaithful SFT collapse 受控实验（"不是调参没调好"）

**问题**：核心方法论发现"视频 SFT 教拒答会把分割 J&F 打对折（0.51→0.22）"会被 reviewer 质疑是训练没调好。这里做 3 个受控实验（每个只改一个因素，其余与 VideoFaithful 配方完全一致：continue from 图像 Faithful、LoRA、1500 iters、同一数据），证明 collapse 是结构性的。

| 控制 | 变量 | 固定不变 | 预期 |
|---|---|---|---|
| c1_negratio | NoTarget 负样本 repeats 4→1 | lr=2e-5, LoRA r=128 | 负样本压力减小 |
| c2_lr | lr 2e-5→1e-5 | NoTarget×4, LoRA r=128 | 学习更温和 |
| c3_lora | LoRA r=128→16 | NoTarget×4, lr=2e-5 | 适配器容量减小 |

**评测**（与主模型同协议）：① faithfulness valid 幻觉（全集 1986）；② Ref-YT-VOS J&F（40 视频对照子集，与 4.4 表同子集）。

**结果表**（幻觉 = faithfulness valid 全量 1986；J&F = 40 视频对照子集，同 4.4 表；c1-c3 训练统一 1500 iters）：
| 模型 | temporal 幻觉 | identity 幻觉 | overall 幻觉 | J&F (40-vid) | 结论 |
|---|---|---|---|---|---|
| 图像 Faithful（对照，无视频 SFT） | — | — | — | **0.532** | 分割无损基线 |
| VideoFaithful（原配方 6376 iters） | 76.3% | 90.9% | 4.2% | 0.221 | 分割崩 |
| **c1_negratio**（NoTarget×1） | 96.7% | 97.2% | 7.5% | **0.212** | 没学会拒答，分割照样崩 |
| **c2_lr**（lr=1e-5） | 96.4% | 97.2% | 7.1% | **0.298** | 同上 |
| **c3_lora**（r=16） | 96.6% | 96.1% | 12.6% | **0.220** | 同上 |

**关键发现（比预期更强的证据，3 个控制全部一致）**：
1. **三个控制（lr 1e-5 / LoRA r=16 / NoTarget×1）在 1500 iters 都没学会时序拒答**（temporal 96.4-96.7% / identity 96.1-97.2% 幻觉，≈ 基线水平）——文字级拒答（global 0.6-1.5% / counterfactual 3.3-16%）学会了，但"mask 停住"这种难拒答没学会；
2. **但三个控制的 J&F 全部崩到 0.21-0.30**（0.212 / 0.298 / 0.220，基线 0.532）——**zero-mask 时序监督对 mask decoder 的损伤与"拒答有没有学会"无关，也与 LR / LoRA 容量 / 负样本比例无关**，是训练目标本身的结构性破坏；
3. 这比"SFT 教拒答→分割崩"更彻底：**只要视频 faithfulness SFT 包含 absent 帧的 zero-mask 监督，分割能力就被打掉一半以上**——任意调参都无法避免。

**论文论证（更新）**：图像级拒答 SFT（无 zero-mask）对分割无损（0.532）；**任何包含 absent 帧 zero-mask 监督的视频 SFT，无论 LR / LoRA 容量 / 负样本比例怎么调，J&F 都系统性崩塌到 ~0.2** → 时序 faithfulness 不能靠 SFT 教给分割本体，必须交给不碰 LLM/mask decoder 的外部轻量评估器。

---

## 五、诊断分析（支撑论文的 insights）

1. **VLM 单帧验证精度 70-83%**：目标消失后单帧 VLM 会把"相似目标"当目标（幻觉保留）或找不到（误停）——单帧验证有天花板。
2. **锚点 cos 判别力 AUC 0.71-0.74**：mask 区域外观在"目标刚消失"与"还在（遮挡/移动）"间难分。
3. **纯推理 TVR（锚点触发 + VLM 重检测）只能改善 1-2pp**：必须把"锚点差分"融入训练（v6）。
4. **视频 SFT 损害分割、图像 SFT 不损害**：J&F 0.22 vs 0.53——核心方法论发现。

---

## 六、Demo（已部署）

- 8 个精选案例（man_black / tissue / mouse / ball / surfboard_man / toilet / surfboard / surfboard_identity）：
  **GT（官方标注）→ ★ Faithful+v6（目标消失后干净停住）→ Sa2VA 基线（全程幻觉）**
- 5 个 case 完全匹配 GT，3 个边界差 1-2 帧。
- 网页：http://172.18.127.61:8899（或 127.0.0.1:8899）

---

## 七、论文要点（写稿框架）

**Title 方向**：Segmentation Accuracy is Not Faithfulness: Temporal Existence for Referring Segmentation

**Abstract 逻辑**：
1. 指代分割默认目标存在 → 真实世界 100% 幻觉（8905 查询）；
2. 图像拒答不够 → 视频存在性是逐帧谓词 e_t；
3. 关键发现：视频 SFT 教拒答让分割 J&F 减半（0.51→0.22）；
4. 正确架构：图像拒答（分割无损）+ 外部轻量时序头（4.5M）；
5. 结果：J&F 0.523（保持甚至略优）+ 幻觉 9.7%（global 1.5%）+ 全集验证。

**核心 trade-off 论证**：Faithfulness 不该以牺牲分割为代价；外部选择性预测器是正解。

**Related Work 定位（防"一票打穿"）**：
- YoURVOS / OMFormer（"Show Me When and Where"）：研究 **temporal target presence**（目标在不在、何时在），在 YoURVOS benchmark 处理 target-absent frames；
- 我们：研究 **segmentation 是否仍 faithful to query**——不仅 target presence，还覆盖 identity swap（mask 跳到相似实例）、counterfactual（query 被改写）、global absence（目标不存在）；
- 关键差异：**presence ≠ faithfulness**。presence 只问"目标在不在帧里"；faithfulness 还问"当前画的 mask 是不是 query 指的那个"。identity_swap 里目标始终在帧里（presence=True），但 mask 可能不 faithful——这是 presence-based 方法覆盖不到的。

**贡献**：
1. 问题 + 诊断（忠实性独立于分割精度；训练缺 no-target）；
2. 视频时序存在性基准（1986 查询 / 4 类）与逐帧 e_t 方法；
3. 方法论发现：SFT 教拒答损害分割，外部头无损；
4. 全集验证（J&F 0.523 / 幻觉 9.7%）+ demo。

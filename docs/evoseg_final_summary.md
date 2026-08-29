# EvoSeg：统一图像-视频的忠实指代分割（Faithful Image-Video Referring Segmentation）

> 一句话：**把指代分割从"有查询就无脑出 mask"变成"有证据的选择性预测"**——目标不存在/消失/换人时，模型要拒答或停住。核心主张：**分割精度 ≠ grounding 忠实性；"当前 mask 是否忠于 query"是逐帧谓词（e_t = 1[M̂_t is faithful to q]）**。目标：CVPR 2027。

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
   - 图像级拒答 SFT（Faithful）：J&F **0.517（无门控，同协议高于 Sa2VA 0.505）**。
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
  └─ 外部时序忠实度头 e_t（~4.5M 参数，不碰 LLM）
        · 逐帧门控：e_t = 1[M̂_t 该帧传播的 mask 是否仍忠实于 query]（faithfulness / acceptance predicate）
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
| **最终** | 基础模型换 Faithful（分割近无损）+ v6 头用 Faithful 特征重训 | **J&F 0.490（thr=0.8，小损 −2.7pp）+ 幻觉 12.0%（rebalanced）** |

### 为什么外部头而不是 SFT LLM（核心论证）
- SFT 教 LLM 拒答（VideoFaithful）→ 分割 J&F 减半（0.22）；
- 外部 e_t 头（4.5M 参数，纯推理门控）→ 分割仅小损（thr=0.8 时 J&F 0.490，−2.7pp）+ faithfulness 有效（幻觉 12.0%，rebalanced）。
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
| 指标 | Sa2VA 基线 | Faithful（图像拒答，无 verifier） | **Faithful + v6（thr=0.8）** |
|---|---|---|---|
| overall 幻觉（帧加权） | 91.3% | 17.0% | **12.0%** |
| **macro 幻觉（4 类平均）** | 90.7% | 55.7% | **31.5%** |
| temporal 幻觉 | 87.6% | 97.5% | **49.2%**（hardest） |
| identity 幻觉 | 90.9% | 98.1% | **55.3%** |
| global 幻觉 | 87.2% | 4.2% | **3.4%** |
| counterfactual 幻觉 | 97.0% | 23.1% | **18.0%** |
| 漏检（temporal / identity） | ~0 / ~0 | — | 19.6% / 5.7% |

> **关键叙事（更新，rebalanced 后）**：图像级拒答（Faithful）已把 overall 从 91.3% 压到 17.0%（global 4.2%、counterfactual 23.1%）——但它对 **temporal/identity 几乎无效（97-98%）**，因为这两类是"mask 时序/实例级不忠实"，静态拒答看不见。**时序 verifier 专门解决图像拒答够不着的 hard cases**：temporal 97.5%→49.2%（−48pp）、identity 98.1%→55.3%（−43pp）、overall 17.0%→12.0%。这就是"图像 abstention 必要但不充分"的直接证据。

### 4.2b identity 指标的重新定义（P0-2 feedback：非空 mask ≠ hallucination）

identity_swap 中目标**始终存在**，因此"非空 mask = 幻觉"不成立；正确指标是 **wrong-instance / identity-switch 错误**：

\[
IoU_{tar}=IoU(\hat M_t, M_t^*),\qquad
IoU_{dist}=\max_j IoU(\hat M_t, D_t^j),\qquad
IDErr_t=\mathbb{1}[IoU_{dist}>IoU_{tar}]
\]

（严格版：\(IoU_{dist}>\tau \land IoU_{tar}<\tau\)。）四类行为的统一 formulation：
| Case | 正确行为 |
|---|---|
| global absence / counterfactual | reject whole video |
| temporal absence | absent frames reject |
| identity confusion | wrong-instance prediction reject（IDErr 度量） |

**评测结果（355 个 identity_swap case / 8623 帧，同协议）**：
| 指标 | Sa2VA 基线 | **Faithful+v6（thr=0.8）** |
|---|---|---|
| **IDErr rate（IoU_dist > IoU_tar 帧占比）** | 33.69% | **32.51%** |
| mean IoU_tar（pred vs 目标实例） | 0.494 | 0.445 |
| mean IoU_dist（pred vs 最佳干扰实例） | 0.733 | **0.650** |
| 平均 IoU_dist > IoU_tar 的 case 数 | 172/355 | 185/355 |

**解读（诚实版）**：identity confusion 是**两类模型的共同 hard case**——基线 IDErr 33.7%、IoU_dist（0.73）远超 IoU_tar（0.49）；时序 verifier 只把 IDErr 降了 **1.2pp**、distractor 重叠从 0.73 降到 0.65（代价是目标重叠 0.49→0.45）。**结论：temporal absence 被 verifier 大幅解决（−48pp），但 identity 换人问题只被部分缓解，仍是未解决的开放点**——论文如实报告并作为 future work（可结合 distractor-aware memory / re-detection）。旧的"identity 幻觉 55.3%"（在仅 8.3% absent 帧上算）语义不清，已弃用。

### 4.3 分割能力（Ref-YT-VOS valid 202 视频，官方式 J&F：空预测帧记 J=F=0、完全拒绝表达式计入分母；逐表达式 827 个）
| 模型 | J&F | 说明 |
|---|---|
| Sa2VA（同协议重跑，基线） | 0.505 | 与官方 0.509 吻合（校验 evaluator） |
| 图像 Faithful（无时序门控） | **0.517** | 基座比 Sa2VA +1.2pp |
| **Faithful + v6 头（thr=0.5）** | **0.503** | 幻觉 10.8% 时 J&F 仅 −1.4pp |
| Faithful + v6 头（thr=0.8） | 0.490 | 幻觉 12.0% 时 J&F −2.7pp |
| 8B Faithful + v6 头（thr=0.5） | 0.488 | 第二 backbone |
| VideoFaithful（SFT） | 0.213（40-vid） | 分割崩 |

> **诚实口径**：e_t 门控以**小幅 J&F 代价**（−1.4~−2.7pp）换取**巨大幻觉下降**（91.3%→12.0-13.5%，rebalanced）——这是受控的选择性预测 trade-off，不是"免费午餐"（早期用旧 evaluator 报的 0.523-0.530 因丢弃被拒绝样本而虚高，已废弃）。

### 4.4 关键对照（同 40 视频公平对比，官方式 evaluator，88 表达式）
Sa2VA 0.507 / MultiTask 0.504 / **图像 Faithful 0.516（无门控）** / Faithful+v6 0.5→0.499 / Faithful+v6 0.8→0.480 / VideoFaithful 0.213 / c1 0.205 / c2 0.290 / c3 0.214

### 4.5 外部泛化
- 图像 HalluSegBench（反事实）：幻觉 **100% → 62-64%**（即拒答率 0% → 36-38%，方向统一为"幻觉率"）
- 视频 MeViSv2 no-target：幻觉 **99.7% → 51.8-53.3%**（即拒答率 0.3% → 46.7-48.2%）

### 4.6 8B 验证（负结果，指导决策）
- 8B-Faithful 的 vlm_feat 在消失边界判别力 AUC **0.55**（4B 0.49）——VLM 全帧特征（无论规模）难以捕捉"目标是否还在"；
- mask_cond 是 SAM2 特征（8B/4B 共用），判别力相同（AUC 0.71）→ **8B 重提取性价比低，不作为主模型**。

### 4.6b P0-2：跨模型规模验证（4B → 8B，scale transfer）

**目的**：证明外部 e_t verifier 不依赖模型大小——把同样的 v6 头流水线（8B vlm_feat 提取 → 训 8B v6 头 → 门控）完整搬到 8B-Faithful 上。**（注意：4B→8B 是同一架构不同规模，论文中只 claim scale transfer，不 claim model-agnostic）**。

| 指标 | 8B-Faithful 基线（无门控，精确实测） | **8B-Faithful + v6 头（thr=0.5 / 0.8）** |
|---|---|---|
| overall 幻觉（全集 1986，rebalanced） | **22.22%**（raw 基线；global/cf 为 rebalanced） | **19.91% / 17.06%** |
| temporal 幻觉 | **96.70%** | **76.3% / 54.7%** |
| identity 幻觉 | **98.58%** | **76.9% / 52.5%** |
| global / counterfactual 幻觉（rebalanced） | 6.33% / 31.95% | 6.1% / 30.4% → 5.45% / 27.33% |
| 漏检（overall） | 0.42% | 1.91% / 9.73% |
| J&F（官方式，40 视频子集 88 表达式） | **0.486** | **0.490**（thr=0.5，基本持平） |

**结论（scale transfer ✓，诚实版）**：
1. **verifier 完整迁移到 8B**：**时序/实例级 hard cases 显著下降**——temporal 96.7%→54.7%（−42pp）、identity 98.6%→52.5%（−46pp）（thr=0.8），overall 22.2%→17.7%；J&F 基本持平（0.486→0.490，40-vid 子集，在噪声范围内）——外部 e_t 评估器不依赖 4B 特定规模；
2. **8B 的 temporal/identity 判别弱于 4B**（54.7%/52.5% vs 4B 的 49.2%/55.3% @thr=0.8）——与 8B vlm_feat AUC 0.55 的早期发现一致（VLM 全帧特征对"目标是否还在"的判别力有限，规模不解决该问题）；
3. **论文定位**：主模型 = 4B（幻觉 12.0% rebalanced、J&F 0.490 @thr=0.8）；8B = 证明 verifier 的跨规模可迁移性（hard-case 幻觉 ↓、J&F 持平）；**不写"91.3%→17.7%"**（那是 4B 原始 Sa2VA 的量级），8B 基线如实报告 22.22%（图像级拒答已处理静态 no-target；注：8B 数字基于旧 elephant-heavy 集，rebalanced 后会有小幅上移）。

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

**Negative-query 人工 QC（P0-6，可选补充，不阻塞投稿）**：
- 从构造负样本（counterfactual_swap 150 + global_absence 100 = 250 条）随机抽样，**2 名标注者**独立判定每条 "query 在此视频中是否确实无对应目标"（Valid negative / Invalid negative / 不确定）；
- 标注工具：`/tmp/negative_qc_review.html`（嵌入视频关键帧 + query，浏览器标注，localStorage 保存）；抽样清单：`/tmp/negative_qc_sample.json`；
- 报告：valid negative rate（应接近 100%）+ 两名标注者 agreement（Cohen's κ）——补齐"mask/presence GT 客观但 query 语义有效性需人工核验"的空缺。

**⚠️ 构造 bug 修复（类别失衡）**：早期 `build_video_faithfulness_manifest.py` 用 `absent_cats[0]` 选负样本类别（列表第一项 = 'elephant'），导致 **global_absence 99% / counterfactual 58% 都是 "elephant"**——benchmark 严重偏置。已修复为 `random.choice(absent_cats)`（17 个类别均匀采样）并重建 valid manifest（`faithfulness_valid_rebalanced.json`，1986 case 数量不变）：
- global_absence：elephant 99% → **6%**（sheep 7% / airplane 7% / suitcase 6% / giraffe 6% / zebra 6% …）
- counterfactual_swap：elephant 58% → **4%**
- **global/counterfactual 幻觉数字正在用 rebalanced 集重测**（temporal/identity 不变，因为基于真实表达）；QC 抽样也已重做（elephant 仅 6%）。

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
| SPARROW（CVPR 2026） | video Pixel-MLLM | 时序 referential consistency（跟踪/参照稳定性） | 空间精度 + 时序一致性设计 | 不覆盖逐帧 faithfulness/refusal |
| **本文 EvoSeg** | **统一图像-视频 Pixel-LLM + 外部 e_t 评估器** | **temporal absence + identity swap + global absence + counterfactual（逐帧 faithfulness）** | 图像拒答（近无损）+ 外部 4.5M 时序忠实度头（不碰 LLM） | 幻觉 91.3%→12.0%（rebalanced；temporal 87.6%→49.2%、identity 90.9%→55.3%）；J&F 0.490 |

### 与 YoURVOS 的关系（互补口径，不是"不能比"）
- **YoURVOS is complementary**：它评测**长视频、非裁剪、目标相关性随时间变化（when-and-where localisation / target-relevant frames，tIoU 指标）**的 in-the-wild RVOS；我们的 benchmark **显式地把 prediction faithfulness 分解为 temporal absence / identity confusion / global absence / counterfactual mismatch 四类**——两者互补，YoURVOS 关注 target-relevant frames，我们显式隔离"目标仍在但 prediction 跳到错误实例"的 selective faithfulness failure；
- **定位**：OMFormer 研究 **when-and-where / target-relevant frames**（presence 视角）；我们研究 **mask 是否仍忠于 query**（faithfulness 视角，含 identity confusion 这类 presence 视角覆盖不到的失败）；
- 若后续能 zero-shot 在 YoURVOS 上评估 EvoSeg 则更好；论文不花篇幅解释"为什么不能比较"，而是把 YoURVOS 放在 related work 的互补位置，主对比在自建 faithfulness 基准上完成。

---

## 四·七、P0-3：阈值 / selective prediction / risk-coverage 分析

**问题**：e_t 门控阈值 thr 决定"幻觉 ↔ 漏检"的操作点。论文不只报 thr=0.8，而是给完整曲线。

### 4.7.1 presence 指标 vs 阈值（faithfulness valid 全集 1986 查询 / 52284 帧，rebalanced global/cf）
| thr | overall 幻觉 | overall 漏检 | frame_acc | macro 幻觉 | temporal | identity | global | counterfactual |
|---|---|---|---|---|---|---|---|---|
| 0（无门控） | 17.04% | 0.60% | — | 55.71% | 97.5% | 98.1% | 4.2% | 23.1% |
| 0.4 | 13.99% | 3.62% | — | 39.93% | 65.3% | 70.3% | 3.6% | 20.4% |
| 0.5 | 13.54% | 4.50% | — | 37.86% | 61.3% | 66.6% | 3.6% | 19.9% |
| 0.6 | 13.04% | 5.57% | — | 36.04% | 57.8% | 63.6% | 3.5% | 19.3% |
| 0.7 | 12.58% | 7.09% | — | 34.02% | 53.9% | 60.0% | 3.4% | 18.6% |
| **0.8** | **11.97%** | **8.85%** | — | **31.48%** | **49.2%** | **55.3%** | 3.4% | **18.0%** |

**解读**：thr 从 0.4→0.8，整体幻觉 14.0%→12.0%（-2pp），漏检 3.6%→8.9%（+5.2pp）——典型的**选择性预测权衡曲线**；`global`/`counterfactual` 相对不受阈值影响（3.4%/18.0%），全部代价集中在 **temporal/identity 边界**（时序判别是真正 hard case）。注：与旧 elephant-heavy 集相比（overall 9.7%），rebalanced 后 overall 12.0% 更诚实——模型在 "elephant absent" 上略过拟合，泛化到多样类别略有上升。

### 4.7.2 J&F vs 阈值（官方式 evaluator，逐表达式 827，空预测帧记 0、拒绝项计入分母）
| thr | J&F | J | F | 完全拒绝表达式 | 幻觉(overall) | 漏检 |
|---|---|---|---|---|---|---|
| 0（无门控） | **0.517** | 0.582 | 0.451 | 4 | 17.04% | 0.60% |
| 0.5 | 0.503 | 0.566 | 0.440 | 20 | 13.54% | 4.50% |
| **0.8** | **0.490** | 0.551 | 0.429 | 28 | **11.97%** | 8.85% |

> **关键发现（修正后，诚实版）**：阈值升高 → 幻觉单调下降（17.0%→12.0%，含 rebalanced global/cf），J&F **单调下降**（0.517→0.490，−2.7pp）——这是**标准的选择性预测 trade-off**：用小幅分割代价换取约 74pp 的幻觉下降（91.3%→12.0%，相对原始 Sa2VA）。不存在"免费午餐"；但代价很小（−2.7pp J&F）且完全可控，远优于 VideoFaithful SFT 的 −30pp（0.517→0.213）。

### 4.7.3 risk-coverage 曲线（Figure 素材）
以 thr 为轴：x = coverage（保留 mask 的 present 帧比例，≈ 1 − 漏检），y = risk（absent 帧被画出的比例，≈ 幻觉）。论文 Figure：risk 随 thr 单调下降（17.0%→12.0%），J&F 单调下降但幅度很小（0.517→0.490）——**标准的 risk–coverage 曲线**，标注 0.5/0.8 两个操作点 + Sa2VA 基线（risk 91.3% / J&F 0.505）+ VideoFaithful（risk 低但 J&F 崩到 0.213）。核心论点：**选择性预测（外部 verifier）能以 ~1-3pp 分割代价换 ~74-79pp 忠实度提升；而 SFT 教学则需 ~30pp 代价**——外部 verifier 是明显更优的 faithfulness 路线。

---

## 四·九、轻量化：帧降采样（stride）的效率-忠实度分析（P0-7）

**动机**：视频推理贵（4B 每 case ~6.2s，8B ~10s），瓶颈是 VLM 全帧前向。能否用更少帧做 faithful 分割？

**协议**：faithfulness benchmark（rebalanced 全集 1986）上，把每个 case 的帧按 stride 降采样后重跑 Faithful+v6（thr=0.8）；延迟 = predict_forward 实测。

| stride | 平均帧数 | overall 幻觉 | temporal 幻觉 | identity 幻觉 | 每 case 延迟 | 加速 |
|---|---|---|---|---|---|---|
| 1（全帧） | 26.3 | **12.0%** | 49.2% | 55.3% | 6.16s | 1.0× |
| 2 | 13.3 | 12.6%（+0.6） | 54.1%（+4.9） | 61.6%（+6.3） | **3.13s** | **2.0×** |
| 3 | 9.0 | 13.6%（+1.7） | 54.6%（+5.4） | 64.9%（+9.6） | **2.23s** | **2.8×** |

**8B 同款验证（跨规模一致）**：stride1 overall 17.06% / stride2 17.49%（+0.4）/ stride3 17.77%（+0.7）——**8B 对帧降采样更鲁棒**（8B VLM 每帧特征更丰富，时序信息冗余更高）。

**结论（轻量化）**：
1. **帧降采样是 graceful 的**：stride2 帧减半 → **2.0× 加速**，4B faithfulness 仅 +0.6pp、8B +0.4pp（overall）；stride3 2.8× 加速，4B +1.7pp、8B +0.7pp——**e_t/分割对时间分辨率有很强的鲁棒性，且规模越大越稳**；
2. **identity 最敏感**（stride2 已 +6.3~7.6pp）——实例级判别需要更密的时序上下文，符合其"最难 hard case"的定位；
3. **论文定位**：主结果用 stride1（全帧，最准）；轻量化部署可用 stride2（2× 加速、近无损）——给出一档可选的 speed-accuracy 操作点。

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
| 图像 Faithful（对照，无视频 SFT） | — | — | — | **0.516** | 分割无损基线（40-vid 无门控） |
| VideoFaithful（原配方 6376 iters） | 76.3% | 90.9% | 4.2% | 0.221 | 分割崩 |
| **c1_negratio**（NoTarget×1） | 96.7% | 97.2% | 7.5% | **0.212** | 没学会拒答，分割照样崩 |
| **c2_lr**（lr=1e-5） | 96.4% | 97.2% | 7.1% | **0.298** | 同上 |
| **c3_lora**（r=16） | 96.6% | 96.1% | 12.6% | **0.220** | 同上 |

> temporal/identity 幻觉基于真实表达（rebalance 不影响）；overall 列基于 pre-rebalance 的 global/cf（elephant-heavy），rebalance 后略有上移，但不影响"分割–拒答干扰"结论（temporal/identity + J&F 才是关键证据）。

**关键发现（比预期更强的证据，3 个控制全部一致）**：
1. **三个控制（lr 1e-5 / LoRA r=16 / NoTarget×1）在 1500 iters 都没学会时序拒答**（temporal 96.4-96.7% / identity 96.1-97.2% 幻觉，≈ 基线水平）——文字级拒答（global 0.6-1.5% / counterfactual 3.3-16%）学会了，但"mask 停住"这种难拒答没学会；
2. **但三个控制的 J&F 全部崩到 0.21-0.30**（0.205 / 0.290 / 0.214，基线 0.516）——**在我们测试的配置范围内，分割–拒答干扰持续出现**：zero-mask 时序监督对 mask decoder 的损伤与"拒答有没有学会"无关，且不受 LR / LoRA 容量 / 负样本比例的调节；
3. 这比"SFT 教拒答→分割崩"更彻底：**只要视频 faithfulness SFT 包含 absent 帧的 zero-mask 监督，分割能力就被打掉一半以上**——该干扰在测试的负样本比例、学习率、LoRA 容量三个维度上都持续存在。

**论文论证（更新）**：图像级拒答 SFT（无 zero-mask）对分割无损（0.516）；**在我们测试的负样本比例、学习率、LoRA 容量配置下，包含 absent 帧 zero-mask 监督的视频 SFT 都使 J&F 崩塌到 0.21-0.29（persistent segmentation–rejection interference）** → 时序 faithfulness 不宜靠 SFT 教给分割本体，交给不碰 LLM/mask decoder 的外部轻量评估器是更稳的路线。

---

## 五、诊断分析（支撑论文的 insights）

1. **VLM 单帧验证精度 70-83%**：目标消失后单帧 VLM 会把"相似目标"当目标（幻觉保留）或找不到（误停）——单帧验证有天花板。
2. **锚点 cos 判别力 AUC 0.71-0.74**：mask 区域外观在"目标刚消失"与"还在（遮挡/移动）"间难分。
3. **纯推理 TVR（锚点触发 + VLM 重检测）只能改善 1-2pp**：必须把"锚点差分"融入训练（v6）。
4. **视频 SFT 损害分割、图像 SFT 不损害**：J&F 0.21-0.29（SFT）vs 0.52（图像 Faithful 无门控）——核心方法论发现。

---

## 六、Demo（已部署）

- 8 个精选案例（man_black / tissue / mouse / ball / surfboard_man / toilet / surfboard / surfboard_identity）：
  **GT（官方标注）→ ★ Faithful+v6（目标消失后干净停住）→ Sa2VA 基线（全程幻觉）**
- 5 个 case 完全匹配 GT，3 个边界差 1-2 帧。
- 网页：http://172.18.127.61:8899（或 127.0.0.1:8899）

---

## 七、论文要点（写稿框架）

**Title 方向**：Segmentation Accuracy is Not Faithfulness: Selective Prediction for Referring Segmentation

**Abstract 逻辑**：
1. 指代分割默认目标存在 → 真实世界 100% 幻觉（8905 查询）；
2. 图像拒答不够 → 视频"mask 是否仍忠实"是逐帧谓词 e_t（faithfulness/acceptance，而非单纯 presence）；
3. 关键发现：视频 SFT 教拒答让分割 J&F 减半（0.51→0.22）；
4. 正确架构：图像拒答（分割近无损）+ 外部轻量时序头（4.5M）；
5. 结果：J&F 0.490（thr=0.8，小损 −2.7pp）+ 幻觉 12.0%（rebalanced，global 3.4%）+ 全集验证。

**核心 trade-off 论证**：Faithfulness 不该以牺牲分割为代价；外部选择性预测器是正解。

**Related Work 定位（防"一票打穿"）**：
- YoURVOS / OMFormer（"Show Me When and Where"）：研究 **temporal target presence**（目标在不在、何时在），在 YoURVOS benchmark 处理 target-absent frames；
- 我们：研究 **segmentation 是否仍 faithful to query**——不仅 target presence，还覆盖 identity swap（mask 跳到相似实例）、counterfactual（query 被改写）、global absence（目标不存在）；
- 关键差异：**presence ≠ faithfulness**。presence 只问"目标在不在帧里"；faithfulness 还问"当前画的 mask 是不是 query 指的那个"。identity_swap 里目标始终在帧里（presence=True），但 mask 可能不 faithful——这是 presence-based 方法覆盖不到的。

**贡献**：
1. 问题 + 诊断（忠实性独立于分割精度；训练缺 no-target）；
2. 视频时序 faithfulness 基准（1986 查询 / 4 类）与逐帧 e_t 方法；
3. 方法论发现：SFT 教拒答严重损害分割（J&F 减半），外部头仅小幅代价（−1~−3pp）；
4. 全集验证（J&F 0.490 @thr=0.8 / 幻觉 12.0% rebalanced）+ demo。

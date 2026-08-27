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
- 图像级 abstention（拒答）前人已做（SESAME 33.5%、GSVA [REJ] 44.6%）；**视频侧的逐帧时序忠实性（e_t）没有端到端解法**——这是本文的差异化点。

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
- 外部 e_t 头（4.5M 参数，纯推理门控）→ 分割无损（0.523）+ 存在性有效（幻觉 9.7%）。
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
| overall 幻觉 | 91.3% | **9.7%** |
| temporal 幻觉 | 87.6% | **49.2%**（hardest） |
| identity 幻觉 | 90.9% | **55.3%** |
| global 幻觉 | 87.2% | **1.5%** |
| counterfactual 幻觉 | 97.0% | **14.9%** |
| 漏检（temporal / identity） | ~0 / ~0 | 19.6% / 5.7% |

### 4.3 分割能力（Ref-YT-VOS valid 202 视频 / 408 表达式，全集 J&F）
| 模型 | J&F |
|---|---|
| Sa2VA 官方 | 0.509 |
| MultiTask | 0.511 |
| 图像 Faithful | 0.526 |
| **Faithful + v6 头（最终）** | **0.523（分割保持，略高于官方）** |
| VideoFaithful（SFT） | 0.221 |
| 旧方案（VideoFaithful + TEG） | 0.246 |

### 4.4 关键对照（同 40 视频公平对比）
Sa2VA 0.514 / MultiTask 0.511 / **Faithful+v6 0.532** / VideoFaithful 0.221 / 旧 TEG 0.246

### 4.5 外部泛化
- 图像 HalluSegBench（反事实）：Sa2VA 100% 幻觉 vs 我们 **拒答 36-38%**
- 视频 MeViSv2 no-target：Sa2VA 99.7% vs 我们 **拒答 46.7-48.2%**

### 4.6 8B 验证（负结果，指导决策）
- 8B-Faithful 的 vlm_feat 在消失边界判别力 AUC **0.55**（4B 0.49）——VLM 全帧特征（无论规模）难以捕捉"目标是否还在"；
- mask_cond 是 SAM2 特征（8B/4B 共用），判别力相同（AUC 0.71）→ **8B 重提取性价比低，不作为主模型**。

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

**贡献**：
1. 问题 + 诊断（忠实性独立于分割精度；训练缺 no-target）；
2. 视频时序存在性基准（1986 查询 / 4 类）与逐帧 e_t 方法；
3. 方法论发现：SFT 教拒答损害分割，外部头无损；
4. 全集验证（J&F 0.523 / 幻觉 9.7%）+ demo。

# EvoSeg — Faithful Image-Video Referring Segmentation
### 论文素材汇总（供 LLM 撰写初稿用，数据截至 2026-08-19）

---

## 0. 一句话定位

> 现有统一图像+视频的 Pixel-LLM（Sa2VA 系）能分割，但不能判断"该不该分割"。它们默认查询目标一定存在，于是面对不存在的目标（no-target）、目标中途消失（temporal absence）、或被相似者顶替（identity swap）时，依然无条件输出 mask。EvoSeg 把指代分割从"无条件的 mask 生成"变成"有证据的选择性预测"，并把 faithfulness 从静态图像假前提扩展到视频时序反事实——全部在**端到端训练**的统一图像+视频模型内完成。

**英文主线（可直接当论文 thesis）**：
> Existing image-video Pixel-LLMs can segment well, but they cannot decide whether segmentation is warranted. EvoSeg turns referring segmentation from unconditional mask generation into evidence-grounded selective prediction, and extends this faithfulness problem from static false premises to temporal counterfactuals in video.

---

## 1. Motivation

- 现在的指代分割（RES/RVOS）评测默认 reference 有效，模型被训练成"有查询就出 mask"。
- 实际场景中查询经常指代**不存在**的东西（用户口误/幻觉/过时信息）、或视频中目标**中途消失/被遮挡/被相似者顶替**——此时正确行为是拒答或停止传播 mask，而不是硬画一个。
- 图像侧已有前人做假前提拒答（SESAME/GSVA/HalluSegBench/VIRO），但：
  1. **视频侧的时序忠实性（目标消失后停止传播、身份切换不跳变）几乎没人做端到端训练**（SPARROW 做一致性但不做 absence 判断；MeViS-Text 挑战赛方案是 test-time/agentic，非端到端；YoURVOS 做 target-absent 帧但非 MLLM 且无拒答语义）。
  2. **没有统一图像+视频的忠实指代分割**。
- 我们的核心主张：**图像级拒答是必要但不充分的——忠实指代分割必须是时序的（existence 是逐帧谓词），模型必须在 referent 消失的那一刻停止 SAM2 传播。**

---

## 2. 关键诊断（论文第一个贡献：empirical finding）

- 在 8905 个 absent query（gRefCOCO 训练集的 no-target 负样本 + COCO 无对应目标查询）上：
  - **Sa2VA-4B：100% 幻觉**（对每个不存在的查询都输出 mask）
  - **4B 多任务 SFT：100% 幻觉**
  - **8B 多任务 SFT：100% 幻觉**
- 同时这些模型正常分割能力很强（RefCOCO ~80 cIoU）→ **分割精度 ≠ grounding 忠实性**。
- **根因（方法层面最有价值的诊断）**：多任务 SFT 的数据构建脚本（`build_pixel_llm_grefcoco.py`）**明确跳过了 no-target 样本**，模型从未被教过"说没有"这个行为 → 后续直接上 RL 也**采不到拒答路径**（RL 无法凭空探索一个训练分布里不存在的行为）。
- 结论：忠实性不是纯 RL 能事后补的能力，必须在训练数据里显式建立。这是"为什么需要 faithfulness SFT 而非纯 RL"的直接证据。

---

## 3. 方法

### Stage 1：图像 faithfulness SFT（教模型"说没有"）

- **配方**：faithfulness-balanced 单阶段 SFT。混合：
  - gRefCOCO no-target 拒答样本 ×4（`Sa2VA07NoTargetDataset`，答案形如 "I don't see X in this image."，无 [SEG]）
  - RefCOCO present ×1（保持分割能力）
  - gRefCOCO present ×1（保持广义指代能力）
- **两个必要基建修复**（可写进 paper 的 implementation details）：
  1. `Sa2VA.forward`：混合 batch 中 0-[SEG]（拒答）样本需补**图连接的零 embedding**（`_zero.expand`，不能 `zeros_like`，否则 DDP 报"未使用参数"）。这让拒答样本参与 mask 头训练、保持 autograd 图一致。
  2. `Sa2VA07NoTargetDataset`：零 mask 输出 + 显式重写 `__getitem__`（绕过多继承 MRO 的坑）。
- **结果**：图像幻觉率 100%→14.7%，gRefCOCO cIoU 29.8%→69.79%，RefCOCO 精度基本不动。

### Stage 2：视频 faithfulness SFT（教模型"目标消失就停"）—— 最差异化贡献

- **数据**：`build_video_faithfulness_train.py` 从 Ref-YT-VOS train 的索引 mask 推导逐帧 presence，生成 **19057 例**：
  - `temporal_absence` 2194 例（referent 只存在于子区间，出现晚/消失早）
  - `identity_swap` 9921 例（视频 ≥2 个同类实例，参照特定实例）
  - `global_absence` 6942 例（跨视频不匹配查询，全部帧无目标）
- **数据集类** `Sa2VA08VideoFaithfulnessDataset`：
  - temporal/identity：真实查询 + "Sure, [SEG]" + 逐帧 GT mask，**消失帧给零 mask**（mask loss 直接教解码器"目标走了就输出空"）
  - global_absence：跨视频不匹配查询 + 拒答文本 + 全零 mask
  - 帧采样保证 presence→absence 边界可见
- **训练**：从 Faithful(×4) 继续，混合 NoTarget×4 + RefCOCO + gRefCOCO + VideoFaithfulness×1，1 epoch。
- **结果**（1986 例视频忠实性基准）：视频整体幻觉 18.4%→**6.6%**，temporal_absence 95%→**76%**，counterfactual 28%→**4.6%**，identity 96%→**81%**，global 3%→**1%**；图像侧 gRefCOCO 69.99%（保持）、RefCOCO 82.24%（保持），幻觉率 16.1%（小回退 1.4pp）。

### Stage 3.5：Temporal Existence Gate（TEG，逐帧存在性门控）—— 新方法

**动机**：图像级拒答 + 视频 SFT 后，帧级 temporal 幻觉降到 76%，但 StopAcc 仅 7.4%（84-93% 的 case 把 mask 传播/闪烁到视频末尾）。根因是单次 [SEG] + SAM2 传播的架构无法表达"逐帧停止"。逐帧重跑模型（N× forward）太贵；且存在性**非单调**（t-1 有 / t 没有 / t+1 又有，遮挡/短暂离场/瞬态状态）。

**TEG 设计（一次前向，几乎零成本）**：
- 保持视频路径：一次前向，视觉编码器并行编码所有帧，LLM 出 [SEG] → SAM2 传播逐帧 mask。
- 新增轻量 **ExistenceHead**：输入 = [SEG] embedding ⊕ 每帧 SAM2 grounding 特征（`current_vision_feats` 空间池化），输出 sigmoid **e_t ∈ [0,1]^T**。
- 训练 loss：分割 loss + **0.5·BCE(e_t, presence_t)**，presence_t 直接从逐帧 GT mask 推导（消失帧为零 mask → presence=0），**零额外标注**。
- 推理：`mask_t = SAM2传播mask_t × (e_t > 0.5)`，逐帧清零。
- 成本：一个小 MLP，和原视频路径同价（非 N× forward）。

**结果（4B，iter 2000，继续自 VideoFaithful）**：

| 指标 | VideoFaithful | **+TEG** |
|---|---|---|
| overall 幻觉 | 6.63% | **6.08%** |
| temporal_absence | 76.3% | **67.5%** |
| StopAcc（干净停住） | 7.4% | **13.9%** |
| never-stop | 92.6% | **84.3%** |
| global_absence | 1.0% | **0.15%** |
| counterfactual_swap | 4.6% | **0.4%** |
| identity_swap | 81.0% | **71.5%** |
| present_miss | 4.2% | 5.5% |

**诚实结论**：TEG 在**全部**忠实性维度带来改善（尤其 counterfactual 4.6→0.4%、identity 81→72%、StopAcc 翻倍），证明逐帧存在性门控有效、能处理非单调存在。但"干净停住"仍未完全解决（StopAcc 14%，84% 仍传播到结尾）——SAM2 传播 + 头锐度仍是瓶颈。可继续：更久训练、提高 existence loss 权重、与 frame-wise 推理结合。

### Stage 3：视频 GRPO RL（reward_temporal_absence）—— 消融/诚实记录

- 写了视频 GRPO 训练器（`grpo_video_train.py`）：
  - reward = 0.7×reward_temporal_presence（present 帧 IoU）+ λ×reward_temporal_absence（**消失帧每多一个 mask 像素扣分**）+ 0.05×format_reward
  - GRPO group-relative advantage + PPO-clip + KL 到 frozen 参考策略
- **工程修复**（可供 paper 附录写）：LLM 输入 max_pixels 压缩（4628→1850 tokens，grad logprob 59GB→30GB）、帧数上限保时序边界、梯度检查点、显存清理。
- **结果**：在 SFT 底座上 RL 无额外增益（overall 6.64% vs 6.63%）。原因：RL 受显存限制用低分辨率 LLM 输入训练、评测用全分辨率，域不匹配。
- **建议叙事（把 negative result 变成 insight）**：
  > **Faithfulness cannot be recovered by post-hoc RL when the refusal trajectory is absent from supervised training.**
  > behavior absent from SFT support ⇒ GRPO exploration fails。
  这个"纯 RL 采不到拒答 trajectory"的观察本身是高级 negative result，比硬包装 Video-GRPO 更值得写。RL 放 ablation，不放进 main contribution。

---

## 4. 实验结果（全部真实数字）

### 4.1 图像侧（RefCOCO/+/g cIoU + gRefCOCO cIoU + 幻觉率）

| 模型 | RefCOCO | RefCOCO+ | RefCOCOg | gRefCOCO cIoU | 幻觉率(8905) |
|---|---|---|---|---|---|
| Sa2VA-4B（基线） | 81.95 | 77.4 | 80.0 | 29.8% | 100% |
| 4B MultiTask SFT | 82.6 | 78.0 | 79.5 | 46.4% | 100% |
| **Faithful-4B（×4）** | 82.22 | 76.49 | 79.02 | **69.79%** | **14.7%** |
| ×6（no-target 占比消融） | 82.22 | 76.99 | 78.53 | 69.12% | 16.63% |
| **VideoFaithful-4B** | 82.24 | **76.77** | **78.51** | **69.99%** | 16.07% |
| **8B Faithful** | 81.44 | 76.52 | 77.31 | **67.37%** | **17.07%** |
| **8B VideoFaithful** | **82.01** | **76.55** | **76.52** | (待测) | 18.07% |

> **RefCOCO+/g 补齐（2026-08-20）**：VideoFaithful-4B RefCOCO+ 76.77 / RefCOCOg 78.51；
> 8B-VideoFaithful RefCOCO+ 76.55 / RefCOCOg 76.52。视频 faithfulness 训练对传统分割几乎无损失
> （4B: RefCOCO 82.24/+/g 76.77/78.51 vs Faithful-4B 82.22/76.49/79.02，g 略降 0.5pp；
> 8B: +76.55/g 76.52 vs 8B-Faithful +76.52/g 77.31，g 略降 0.8pp）。

> 口径：gRefCOCO cIoU 是"分割+拒答"复合指标（absent 查询正确拒答记 1.0）。Sa2VA 公开数字：4B 82.4/77.6/79.7，8B 82.6/78.0/80.3。gRefCOCO SOTA 参考：Text4Seg ~70，GSVA ~65（我们不宣称 SOTA，定位为"Sa2VA-style 统一模型内 fidelity 大幅提升且不损精度"）。
> **8B VideoFaithful 图像侧（2026-08-20 补测）**：absent 幻觉率 18.1%（8B Faithful 17.1%→18.1%，
> 视频 faithfulness 训练对图像拒答有 ~1pp 小回退，与 4B 的 14.7%→16.1% 模式一致）。8B 视频训练
> 主要收益在视频侧（overall 4.96%、temporal 61.1%），图像侧 8B 略逊 4B（18.1% vs 14.7%）。
> **HalluSegBench 外部反事实泛化（50 对）**：8B VideoFaithful factual 0.98 / 反事实拒答 36%（与 4B 的 36-38% 持平）。

### 4.2 视频忠实性基准（1986 例 / 52284 帧，absent_halluc_rate）

| 模型 | overall | temporal_absence | global_absence | counterfactual_swap | identity_swap |
|---|---|---|---|---|---|
| Sa2VA-4B | 91.3% | 87.6% | 87.2% | 97.0% | 90.9% |
| Faithful-4B | 18.4% | 95.3% | 3.1% | 28.1% | 96.4% |
| **VideoFaithful-4B** | **6.6%** | **76.3%** | **1.0%** | **4.6%** | **81.0%** |
| VideoFaithful-4B + RL | 6.6% | 76.8% | 1.0% | 4.6% | 81.0% |
| 8B Faithful（仅图像） | 23.7% | 92.4% | 4.3% | 39.6% | 98.6% |
| **8B VideoFaithful** | **5.0%** | **61.1%** | **0.7%** | **2.8%** | **70.2%** |

> **8B VideoFaithful（iter16984，gpu8 训练完成 + 转 HF + 视频评测完成，2026-08-20）**：
> 全维度优于 4B VideoFaithful（overall 4.96% vs 6.63%，temporal 61.1% vs 76.3%，
> identity 70.2% vs 81.0%，StopAcc 15.7% vs 7.4%，never-stop 74.1% vs 92.6%）。
> 代价：present_miss 更高（overall 8.3% vs 4B 4.2%，temporal 20.8% vs 9.8%）——8B 更保守
> （"宁可少画也不错画"），是显式取舍，可作为 model scale 讨论。
> 结论：**视频时序忠实性随模型规模提升**——8B 在 hardest 的 temporal/identity 上比 4B 各降 ~15pp / ~11pp。

> 补充：VideoFaithful-4B 的 present_miss_rate = 4.2%（overall），temporal 类别 9.8%——模型更保守，是取舍。frame_acc overall 93.9%。
> **时序错误分解（GPT 评审预判的关键补充，StopAcc/StopLatency）**：temporal_absence 的 108 个 disappear_early 案例上：
> - StopAcc（边界后干净停止的案例比例）：Sa2VA 2.8% / Faithful 1.9% / **VideoFaithful-4B 7.4%** / **8B VideoFaithful 15.7%**
> - never-stop（mask 传播/复现到视频末尾的案例比例）：Sa2VA 88.9% / Faithful 96.3% / **VideoFaithful-4B 92.6%** / **8B VideoFaithful 74.1%**
> - 8B VideoFaithful 的 mean_stop_latency 5.7 帧、mask_leakage 3.9%
> - **结论：temporal 幻觉不是"消失后一帧残留"，而是 mask 持续传播/闪烁复现到结尾**。帧级改善是真实的（4B：95%→76%；8B：→61%），且**随规模扩大 StopAcc 7.4%→15.7%、never-stop 92.6%→74.1%**——但"干净停住"在单 [SEG] 决策 + SAM2 传播架构下仍不彻底，这是架构极限的诚实证据，恰恰支撑"忠实指代分割必须时序化 / 需要逐帧存在性验证"的主线。

### 4.3 消融

| 消融 | 结论 |
|---|---|
| no-target 占比 ×4 vs ×6 | ×6 无增益（幻觉 16.6% vs 14.7%，gRefCOCO 69.1 vs 69.8）→ ×4 是平衡点 |
| 视频 GRPO RL vs SFT | 无额外增益（6.64% vs 6.63%）→ SFT 是主机制 |
| 纯 RL（无 no-target 训练数据） | 推不动（采样不到拒答路径）→ 数据是根因 |

---

## 5. Related Work 定位（重要：不要写"首次解决 no-target"，会翻车）

**已占坑（必须在 related work 里正确引用并差异化）**：
- **GRES/gRefCOCO**（CVPR'23）：定义 single/multi/no-target。
- **GSVA**（CVPR'24）：专门的 [REJ] token 做 empty-target rejection。
- **SESAME "See, Say, Segment"**（CVPR'24）：false-premise 检测 + 自然语言反馈 + 纠正 + FP-RefCOCO/+/g 基准。
- **HalluSegBench**（2025）：pixel-grounding hallucination、factual/counterfactual pairs、CMS/CCMS 指标。
- **VIRO**（CVPR'26）：REC 神经符号验证 + abstain + no-target，balanced 61.1%。
- **SPARROW**（CVPR'26）：视频 MLLM 时序指代一致性（TSF + 双 prompt），做 identity switches/move/reappear 但**不做 absence 判断**。
- **YoURVOS/OMFormer**（2026）：未裁剪视频 RVOS，target-absent 帧基准，但非 MLLM、无拒答语义、无图像统一。
- **MeViS-Text 挑战赛（PVUW'26）**：正式纳入 no-target 评测（N-acc/T-acc），但方案是 agentic/test-time 验证，非端到端训练。MeViSv2（TPAMI）：3503 条 no-target 标注。
- **Learning to Refuse / RA-RFT**（CVPR'26）：GRPO 拒答 RL，但是 Video Temporal Grounding（文本时序），非像素级。
- **LENS / SAM-R1 / Seg-Zero / SAMTok / StAR**：GRPO+分割 reward（我们不做 method novelty 主张在这块）。
- **CFCamo**（2026）：counterfactual paired reward（伪装检测，box 层面）。
- **InstructSAM / Inst²Seg**：single/multi/no-target 指令分割基准。

**我们的可防守差异化（写进 contribution）**：
1. 统一图像+视频 Pixel-LLM 内，**端到端**学会逐帧存在性判断 + 拒答 + 停止传播（不是 test-time agentic、不是后处理 gating、不是文本/时序区间）。
2. **时序反事实 benchmark**：1986 例 / 52284 帧，四类（temporal_absence / global_absence / counterfactual_swap / identity_swap），专门打"目标消失后 SAM2 仍传播 mask"。
3. **诊断发现**：所有 SFT 模型对 absent query 100% 幻觉 + 根因是训练数据缺 no-target → "图像级 abstention 必要但不充分，忠实指代必须时序化"。
4. 诚实消融：no-target 占比敏感性、RL vs SFT。

---

## 5.5 外部基线（GPT 清单第 2、3 项，已完成）

**外部模型对照（证明非 Sa2VA-specific）**——SESAME（CVPR'24，专为 false-premise 拒答训练，LLaVA-7B）在我们同款 8905 absent 查询上：

| 模型 | 幻觉率(8905 no-target) |
|---|---|
| Sa2VA-4B | 100% |
| **SESAME** | **33.5%** |
| **GSVA-7B（[REJ] token）** | **44.6%** |
| **Faithful-4B（我们）** | **14.7%** |
| VideoFaithful-4B | 16.1% |

> **GSVA 补跑完成（2026-08-20）**：官方 gsva-7b-ft-gres.bin（gRefCOCO 微调、显式 [REJ] 拒答 token），
> 修复了权重前缀 + LoRA(r=8,α=16) 合并后跑通（正样本→[SEG]、负样本→[REJ] 校验通过）。
> 在 8905 absent 查询上幻觉 **44.6%**（3968/8905），与它官方 gRefCOCO N_acc≈0.57 吻合。
> 结论：**有显式 [REJ] token 的 GSVA 也会幻觉 44.6%**——比 SESAME(33.5%) 还高，比我们 Faithful-4B(14.7%) 高 3 倍。
> 这进一步坐实"非 Sa2VA-specific"且**我们的拒答能力优于专用假前提/拒答模型**（SESAME + GSVA 都是 LLaVA-7B 级，
> 我们同样 4B 量级但幻觉率低一半以上）。（口径：外部模型用各自原生 prompt/格式，非严格同配，作参照。）

**外部基准泛化（HalluSegBench，反事实）**——test refer_seg 50 对 factual/counterfactual：

| 模型 | factual 分割率 | counterfactual 幻觉率 | 拒答率 |
|---|---|---|---|
| Sa2VA-4B | 1.0 | **100%** | 0 |
| VideoFaithful | 0.98 | **64%** | 36% |
| TEG | 0.98 | **62%** | 38% |

> 结论：外部反事实基准上 Sa2VA 100% 幻觉，我们的模型在**未见过的外部数据**上拒答 36-38%（反事实把目标换成相似物，本身有歧义，所以比自有 8905 高）。

## 6. 投稿目标与定位建议

- **目标**：CVPR（主会）。当前完整度按此前评审模拟评估约为 Weak Accept 区间。
- **不要宣称**：❌ 首次解决 no-target / 首次用 RL 优化分割（前人已占）。
- **正确宣称**：图像级 abstention 必要但不充分；把 faithfulness 从静态假前提推广到视频时序反事实；统一图像+视频端到端忠实指代分割。

---

## 7. 论文结构建议

1. **Abstract**：motivation（能分割 ≠ 知道该不该分割）→ 诊断（100% 幻觉 + 根因）→ 方法（Stage1 图像 faithful SFT + Stage2 视频 faithful SFT）→ 结果（图像 14.7% 幻觉、gRefCOCO 69.8；视频整体 6.6%、temporal 76%）→ 贡献 4 条。
2. **Introduction**：Figure 1 = 不同模型/策略在 8905 absent query 上 100% 幻觉 + 分割精度 ~80 的对比 + 三张可视化（图像假前提 / 视频消失后传播 / 反事实换 query）。诊断先行，然后方法概览，贡献列表。
3. **Related Work**：GRES/GSVA/SESAME/HalluSegBench/VIRO（图像假前提）；SPARROW/YoURVOS/MeViS-Text/RA-RFT（视频与时序）；LENS/SAM-R1（RL 分割）；明确边界。
4. **Method**：
   - 3.1 Problem formulation：e = 1[∃o: q(o)=1]，e_t 逐帧存在性，M_t = f(V,q,t) if e_t else ∅。
   - 3.2 诊断（训练分布缺失 no-target → RL 不可达）。
   - 3.3 图像 faithfulness SFT（balanced mix + forward zero-embedding fix + dataset fix）。
   - 3.4 视频 faithfulness SFT（manifest 构建 + 零 mask 消失帧 + 跨视频负样本 + 帧采样保边界）。
   - 3.5（可选 ablation 章节）视频 GRPO RL 与 reward_temporal_absence。
5. **Experiments**：4.1 设置（数据/基准/指标口径）；4.2 图像主表；4.3 视频忠实性主表（before/after：Sa2VA→Faithful→VideoFaithful）；4.4 消融（×4 vs ×6、SFT vs RL）；4.5 定性可视化 + 失败案例分析。
6. **Conclusion**：图像 abstention 必要不充分；时序 faithful 是下一块拼图；局限（present_miss 上升、temporal 仍 76%、identity 仍 81%）。

---

## 7.5 评审预判与应对（GPT 2026-08 评估：CVPR 2027 Weak Accept → 补完关键实验后 competitive）

**评分参考**：Problem importance 8.5 / Story-insight 8.5 / Novelty 7.5 / Method novelty 6.5 / Evidence 7 / CVPR fit 9 / ICLR fit 6.5-7。优先投 **CVPR**（ICLR 2027 deadline 2026-09-25 太近且本工作 method novelty 对 ICLR 不够）。

**必须堵的三个 reviewer 攻击点**：
1. **"是不是 Sa2VA-specific pathology？"** → 补外部模型基线（GSVA 的 [REJ]、SESAME 至少覆盖两类，不能全是自己的 checkpoint）。
2. **"benchmark 是不是和训练分布太近？"** → 补 external benchmark generalization：图像用 FP-RefCOCO / HalluSegBench，视频用 MeViSv2 no-target / YoURVOS。
3. **"overall 6.6% 是不是被 easy negative 拉下来的？temporal 76% 怎么解释？"** → 用上面的 StopAcc/StopLatency/MaskLeakage 分解 + error taxonomy（near-miss / long-tail / ambiguous identity）+ 诚实承认 temporal 是开放问题并指向架构极限。

**不要写的**：❌ 首次解决 no-target / 首次 RL 分割。**要写的概念链**：
```
Segmentation accuracy is not faithfulness.
Image faithfulness ≠ Temporal faithfulness.
e (existence)  ⟶  e_t (frame-wise existence).
```

## 8. 待办/未完成（写 paper 前可补，不阻塞初稿）

- [x] **失败分类学**：剩余图像幻觉 75% 是 far-miss（查询类别在图中根本不存在）、25% 是 lookalike
      near-miss（类别在但实例/属性/位置不符）；且残留幻觉全部是自信 [SEG]。
      → `projects/evoseg/eval/failure_taxonomy.py`
- [x] **Figure 1 可视化**：`make_figure_cases.py` + `make_figure1.py` → `evo_artifacts/figures/figure1.png`
- [x] 8B VideoFaithful 训练（gpu8, iter16984 完成）+ 转 HF（EvoSeg-Qwen3-VL-8B-VideoFaithful）
      + 视频评测（overall 4.96% / temporal 61.1% / identity 70.2% / StopAcc 15.7%）
      + 图像 absent 幻觉（18.1%）+ HalluSegBench（拒答 36%）
- [x] VideoFaithful-4B 的 RefCOCO+/g（76.77/78.51）与 8B-VideoFaithful 的 RefCOCO 82.01 / + 76.55 / g 76.52
- [x] 基线表：SESAME(33.5%) / **GSVA(44.6%)** / HalluSegBench 已跑；Text4Seg 待补（可选）
- [x] 图像外部泛化：HalluSegBench（Sa2VA 100% vs 我们 36-38% 拒答）
- [ ] 视频外部泛化：**MeViSv2 no-target / YoURVOS**（下一个，需下载完整数据集 + 构造 no-target 查询）
- [x] 外部模型基线：SESAME 已跑（33.5% vs 我们 14.7%）；HalluSegBench 已跑（Sa2VA 100% vs
      VideoFaithful/TEG 拒答 36-38%）；**GSVA 已跑（44.6% vs 我们 14.7%）**
- [x] 升级后的 eval_video_faithfulness.py 已输出 StopAcc/StopLatency/MaskLeakage（temporal_stop 指标）
- [ ] 置信度校准 / risk-coverage（可选加分项）

**核心闭环已完整（2026-08-20）**：诊断（100% 幻觉 + 数据根因）→ 图像拒答（100%→14.7%）→
视频时序化（91.3%→5.0%）→ 外部基线（SESAME 33.5% / GSVA 44.6% / HalluSegBench 36-38% 拒答）→
失败分类学 + Figure 1 → 8B 规模验证。剩下可选的：MeViSv2/YoURVOS 视频外部泛化、Text4Seg、置信度校准。

### 失败分类学结果（failure analysis 小节素材）

剩余图像幻觉（Faithful-4B, 1309/8905）按"查询结构 × 接地难度"分解：

| 结构 \ 难度 | far-miss（类别不存在） | lookalike（类别在，实例/属性不符） |
|---|---:|---:|
| attr_rich（>3 词） | 671 (51.3%) | 282 (21.5%) |
| short_noun（≤3 词） | 314 (24.0%) | 42 (3.2%) |
| **合计** | 985 (**75.2%**) | 324 (**24.8%**) |

- 对比 Sa2VA-4B（8905/8905 幻觉）：lookalike 仅 10.1% → Faithful 升到 24.8%。训练后被拒掉的
  主要是"清楚的假前提"，**残留幻觉集中在 genuinely hard 的 near-miss**（类别在、但所指
  实例/属性/位置不符），比"什么都画"健康得多——可直接作为 failure analysis 的诚实叙事。
- 全部 1309 条残留幻觉的 pred_text 都是自信 "Sure, [SEG]."，没有"文本拒答 + mask"的语义混淆。

### Figure 1（opening figure）材料

- 复现：`make_figure_cases.py`（GPU 上跑 Sa2VA/Faithful/TEG，出 case 图）+
  `make_figure1.py`（matplotlib 拼版）。产物 `evo_artifacts/figures/figure1.png` (3823×1960)。
- (a) 双模态柱状图：图像 absent 幻觉 100%→14.7%、视频 absent 幻觉 91.3%→6.1%，
  RefCOCO 81.95→82.22 / gRefCOCO 29.8→69.8 保持并提升。
- (b) 图像 case：`COCO_train2014_000000274667.jpg` + "the red jacket"（不存在）。
  Sa2VA 输出 "Sure, [SEG]." + 占 68.7% 画面的红色幻觉 mask；EvoSeg-4B 输出
  "I don't see red jacket in this image." + 无 mask。
- (c) 视频 case：Ref-YT-VOS valid `0788b4033d` + "a man walkng in an all black outfit"，
  目标在第 12 帧消失。Sa2VA 单次 [SEG]+SAM2 传播：`1111111111111111111`（mask 画到末尾）；
  EvoSeg-4B 逐帧 e_t：`1111111111110000000`（第 12 帧干净停住）。
  → "忠实指代分割必须时序化（e → e_t）"最直观的视觉证据。

---

## 9. 复现信息（代码/数据/命令）

- **仓库**：`/9950backfile/chenjiahui/EvoSeg`（GitHub: cjhcjh123-666/EvoSeg，分支 master；最近 commit `2a1f3eb`）
- **Python**：`projects/sa2va/.venv/bin/python`（torch 2.6, xtuner 0.1.23, peft 0.17）
- **关键脚本**：
  - 数据：`projects/evoseg/tools/build_video_faithfulness_train.py`、`build_no_target_abstain.py`
  - 数据集：`projects/sa2va/datasets/sa2va_data_07_notarget.py`、`sa2va_data_08_video_faithfulness.py`
  - 配置：`projects/sa2va/configs/evoseg/sa2va_qwen3_{4b,8b}_{notarget,video_faithfulness}_sft.py`
  - 评测：`projects/evoseg/eval/eval_no_object_halluc.py`（8905）、`eval_video_faithfulness.py`（1986）、`projects/sa2va/evaluation/sa2va_eval_refcoco.py`
  - RL：`projects/evoseg/rl/grpo_video_train.py`、`grpo_fuse_eval.py`、`rewards.py`
- **数据**（`/9950backfile/chenjiahui/evo_artifacts/datasets/`）：gRefCOCO、coco2014、ref_youtube_vos、mevis_v2、s4b/pixel_llm_data
- **训练命令**：
  ```
  torchrun --nproc_per_node=8 tools/train.py <config> --launcher pytorch --deepspeed ""
  ```
- **转换 HF**：`python tools/convert_to_hf.py <config> <pth> --save-path <out>`
- **模型目录**：`evo_artifacts/models/EvoSeg-Qwen3-VL-4B-{MultiTask,Faithful,VideoFaithful,VideoFaithful-RL}`、`EvoSeg-Qwen3-VL-8B-Faithful`
- **权重/数据不入库**（AGENTS.md 约定），实验结果在 `evo_artifacts/results/s4b/eval/`

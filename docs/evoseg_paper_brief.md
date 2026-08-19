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
| **VideoFaithful-4B** | 82.24 | (待补) | (待补) | **69.99%** | 16.07% |
| **8B Faithful** | 81.44 | 76.52 | 77.31 | **67.37%** | **17.07%** |

> 口径：gRefCOCO cIoU 是"分割+拒答"复合指标（absent 查询正确拒答记 1.0）。Sa2VA 公开数字：4B 82.4/77.6/79.7，8B 82.6/78.0/80.3。gRefCOCO SOTA 参考：Text4Seg ~70，GSVA ~65（我们不宣称 SOTA，定位为"Sa2VA-style 统一模型内 fidelity 大幅提升且不损精度"）。

### 4.2 视频忠实性基准（1986 例 / 52284 帧，absent_halluc_rate）

| 模型 | overall | temporal_absence | global_absence | counterfactual_swap | identity_swap |
|---|---|---|---|---|---|
| Sa2VA-4B | 91.3% | 87.6% | 87.2% | 97.0% | 90.9% |
| Faithful-4B | 18.4% | 95.3% | 3.1% | 28.1% | 96.4% |
| **VideoFaithful-4B** | **6.6%** | **76.3%** | **1.0%** | **4.6%** | **81.0%** |
| VideoFaithful-4B + RL | 6.6% | 76.8% | 1.0% | 4.6% | 81.0% |
| 8B Faithful（仅图像） | 23.7% | 92.4% | 4.3% | 39.6% | 98.6% |
| 8B VideoFaithful | 训练中（ETA 2026-08-19 ~14:40） | | | | |

> 补充：VideoFaithful-4B 的 present_miss_rate = 4.2%（overall），temporal 类别 9.8%——模型更保守，是取舍。frame_acc overall 93.9%。
> **时序错误分解（GPT 评审预判的关键补充，StopAcc/StopLatency）**：temporal_absence 的 108 个 disappear_early 案例上：
> - StopAcc（边界后干净停止的案例比例）：Sa2VA 2.8% / Faithful 1.9% / **VideoFaithful 7.4%** / 8B 1.9%
> - never-stop（mask 传播/复现到视频末尾的案例比例）：Sa2VA 88.9% / Faithful 96.3% / **VideoFaithful 92.6%** / 8B 84.3%
> - 5 帧内出现停止的案例：Sa2VA 39.8% / Faithful 37.0% / **VideoFaithful 41.7%**
> - **结论：76% 的 temporal 幻觉不是"消失后一帧残留"，而是 mask 持续传播/闪烁复现到结尾**。帧级改善是真实的（95%→76%，5 帧内停止 37%→42%），但"干净停住"在单 [SEG] 决策 + SAM2 传播架构下几乎做不到（StopAcc 仅 7.4%）——这是架构极限的诚实证据，恰恰支撑"忠实指代分割必须时序化 / 需要逐帧存在性验证"的主线。
> 8B 视频列明显弱于 4B，因为 8B 目前只做了图像 no-target SFT、未做视频 faithfulness SFT（正在补）。

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

- [ ] 8B VideoFaithful 训练（进行中）+ 转 HF + 视频评测（~3.5h）
- [ ] VideoFaithful-4B 的 RefCOCO+/g（~40min）
- [ ] 基线表：SESAME / GSVA / Text4Seg / HalluSegBench（需跑外部模型，~0.5-1 天）
- [ ] Figure 1 可视化（100%→6.6% + 3 张 case，~1-2 天）
- [ ] 跨数据集泛化：FP-RefCOCO / HalluSegBench（图像）、MeViSv2 no-target / YoURVOS（视频）（~0.5-1 天，需下载数据）
- [ ] 外部模型基线：GSVA（[REJ]）、SESAME 至少两类（~1 天，需下载/搭 LISA-based 环境）——**GPT 认为最重要的一项**
- [ ] 用升级后的 eval_video_faithfulness.py 重跑全部模型，补 StopAcc/StopLatency/MaskLeakage 列
- [ ] 错误分类学：剩余 1309 条图像幻觉 + 视频 temporal/identity 失败是 near-miss 还是 far-miss
- [ ] 置信度校准 / risk-coverage（可选加分项）

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

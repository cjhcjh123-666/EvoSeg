# 分割幻觉的区域证据实验

当前 EvoSeg 研究图像分割 VLM 的像素 grounding 幻觉，训练和评测继续使用公开数据。新 pilot 检验一个具体假设：保留完整的语言推理表示，以模型自己预测的 mask 为条件读取区域与上下文的视觉证据，可以改善视觉错配，同时避免继续更新语言模型造成的误拒绝。

## 已有证据

100 步模型在完整 gRefCOCO val 上达到 gIoU 75.43，相比发布基模 70.65、同条件 full-view 对照 72.08 有收益。HalluSegBench 指代子集的文本错配 CMS 从 0.3047 降至 0.1408，但视觉反事实 CMS 只从 0.6690 降至 0.6389；推理子集改善有限。

继续到 300 步时，507 条有目标训练留出表达的误空结果从 5 增至 17，其中 15 条是语言模型直接输出空结果。相同的 490 条始终非空样本 IoU 下降约 1.21 点。新增误拒绝贡献约 58% 的正样本 IoU 净损失，同条件对照同样退化。这支持检查共享训练策略，但不能证明所有退化都来自拒绝偏置。

## 相关方法与边界

[RobustSeg](https://arxiv.org/html/2506.21546v3) 使用 factual/counterfactual 图文四元组联合训练正目标和空目标分割。其方向直接覆盖视觉幻觉，但本轮不使用 HalluSegBench 编辑图像训练。

[PropVG](https://openaccess.thecvf.com/content/ICCV2025/html/Dai_PropVG_End-to-End_Proposal-Driven_Visual_Grounding_with_Multi-Granularity_Discrimination_ICCV_2025_paper.html) 使用句子与词级对比 refer scoring，以及对象和语义级信息融合来判断缺失目标。[InstAlign](https://arxiv.org/html/2411.15087) 显式对齐对象实例与语言短语，以 relevance 聚合分割和无目标结果。因此，区域匹配、置信度头和拒绝头本身不是新贡献。

[CoReS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02711.pdf) 通过推理和分割双链逐步细化视觉搜索。这里吸收“推理表示应辅助定位”的原则，不引入生成式 reasoning-chain 标签或新的训练文本。

EvoSeg 的实验区别是：在同一 Seg-VLM 内保留完整 semantic 表示和原有 interaction 表示，将其与预测 mask 内部、外部的视觉 token 特征共同用于一个有界 prompt residual。区域与上下文的分解能否支持新的 scientific claim，取决于与不做区域池化的同容量对照及视觉反事实结果；目前不宣称首次提出区域 grounding 或因果识别。

## 实现与监督

固定 100 步的 Qwen3-VL-8B-SAMTok、语言 LoRA、原 grounding adapter 和 SAM2.1。新模块只训练区域条件残差及候选支持分数。语言生成保持不变，不能通过把更多有目标表达改成空回答来降低幻觉。

候选 mask 只能由冻结模型从原图、原 query 生成，不能由 GT mask 定义区域输入。区域特征从同一次 generation prefill 的 image tokens 提取，验证 merged token grid 后按原生 mask 阈值 `logits > 0.5` 池化。语言完整表示不被全部扣除，以保留隐式指代需要的知识与上下文。

训练监督仍为原始公开 human mask 和官方 no-target 标签。候选的 soft support target 是其前景与原始 GT 的交叠精度，忽略 ReasonSeg 的 ignore 区域；这是一项由人工 mask 计算的训练目标，不是模型预测生成的伪 GT。采用 precision 而不是 union IoU，避免把合法的单实例 mask 因多目标 query 的其他实例而标为不支持。

像素 BCE、Dice、候选支持 BCE 和正样本残差保持项共同训练；修正幅度限制在 parent prompt RMS 的 10%。初始残差为零，支持门槛固定为 0.5，初始支持偏置为 2。冻结模型不产生候选的表达仍输出空结果，因此本 pilot 不解决原有语言层漏检。

## 对照与验收

regional 与 global 从同一 parent 开始，使用相同原始样本、候选 mask、监督、参数量和更新次数。global 仍计算区域池化，但将两个区域输入替换为全图均值，并移除显式面积输入，以隔离 mask 条件视觉证据的作用。两个变体均保留原 segmentation prompt，因此 global 并非完全没有空间信息。

先缓存 1024 条公开训练表达及全部 695 条图像隔离训练留出表达。缓存覆盖包括语言直接输出空结果的记录；训练中只有候选存在的记录进入新接口，排除数量和来源单独报告。公开 benchmark 不过滤这些表达。

pilot 通过真实梯度与原生推理检查后，再按训练留出集检查正常分割和召回。公开 HalluSegBench 与 gRefCOCO 按完整划分评测，不能用公开分数选权重或阈值。目标是视觉反事实与推理子集均改善，且正常分割和召回没有明显下降；执行完成不等于满足验收，更不等于 SOTA。

现有长训及公开评测队列保留。区域特征缓存只在 GPU 进程全部属于本项目只读公开评测、且每卡至少有 55 GB 可用显存时并行；不与语言 LoRA 训练共享该缓存阶段。轻量训练和私有留出集检查使用冻结特征，不重新更新语言模型。私有缓存推理仅消费模型特征和 SAM 图像状态，人工 GT 仅在预测之后用于计算指标；它不作为公开 benchmark 推理路径。

区域实验的完整公开评测仍排在现有公开队列之后，采用原生图像和 query 重新推理。所有工作仍以北京时间 2026 年 10 月 7 日 22:00 为截止；来不及完成的完整评测须明确标为待完成。

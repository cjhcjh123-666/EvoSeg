# VIRST 机制审计（positive control）

## 审计范围

- 官方仓库：`https://github.com/AIDASLab/VIRST`
- 本地官方 checkout：`/9950backfile/chenjiahui/evo_artifacts/external/VIRST`
- commit：`00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`
- 论文：[arXiv:2603.27060](https://arxiv.org/abs/2603.27060)，v1 提交于 2026-03-28（CVPR 2026）；本轮同时核对 arXiv 摘要与官方实现。
- 本轮只读代码和论文/README，不重新训练或推理 VIRST。

## 事实核对

论文摘要把两个相关组件定义为：Spatio-Temporal Fusion（把 segmentation-aware video features 融入 VLM backbone）以及 Temporal Dynamic Anchor Updater（在大运动、遮挡和重现时维护时间相邻的 anchors）。以下逐项核对公开代码中可观察到的对应数据流；摘要中的 SOTA/泛化主张不在本轮复验范围内。

1. **时序特征在 object/pixel prompt 形成前进入。** `model/VIRST.py:260-278` 先从所有 `T_seg` 帧抽取 SAM2 vision features，再由 `InitialSegFusion` 把这些时空视觉 token cross-attend 到 `[SEG]` 输入 embedding。`model/seg_prompter.py:316-338` 显示该融合读取展平后的 `T × spatial` memory，并以残差门控写回 segmentation token。

2. **不是一个全视频共享的单一 pixel prompt。** `model/seg_prompter.py:251-282` 将 query 扩展到每个时间位置，经两层带时空 RoPE 的 decoder 输出 `(conversation, SEG token, T, 256)` 的逐帧 prompt，并由最后一层 attention 聚合得到 frame score。

3. **object selection / keyframe selection 发生在 pixel execution 前。** `model/VIRST.py:342-380` 先由 `SegPrompter` 产生逐帧 embeddings 和 attention scores，再据此选 conditioning frames；因此多帧信息并非等到完整 track 产生后才参与选择。

4. **frame-specific state 直接进入像素执行器。** `model/VIRST.py:484-524` 对每个 conditioning frame 调用 `seg_model.add_new_prompt`，传入该帧自己的 learned 256-D prompt，然后才调用 `propagate_in_video`。这不同于 Sa2VA 的 single/global `z_seg` 经一次 prompt 后整段传播。

5. **tracking 维护对象身份和视频 memory。** `model/sam2/sam2_virst.py:208-270` 将同一 `obj_id` 的 prompt 写入对应帧状态；已有输出可作为后续 correction 的 mask logits。VIRST 的 evaluation path 对多个 anchor 写入同一对象，再由 SAM2 video memory 传播。

## 不能由审计单独推出的结论

- 代码支持“时序特征在 pixel prompt 前融合、并形成多 anchor 的 frame-specific prompt”这一事实。
- 它**不是**观察上一阶段预测后再在线重新理解 query 的 agent/update loop；多个 prompt 是一次模型计算后共同得到的。
- VIRST 的 Dynamic–Static gap 约为 `-0.37 pp` 是既有实测 positive control，但和 Sa2VA 在 backbone、训练数据、参数量及 pixel executor 上均有混杂。仅凭这次代码审计，不能把 gap 缩小因果归结为某一个模块。

## 对本轮诊断的用途

VIRST 提供的是架构层面的 positive control：temporal fusion 同时位于 segmentation-token 形成之前和 pixel prompt 形成过程中，并输出 frame-specific prompts。是否必须采用同类结构，仍由 Oracle Ladder 和 frozen diagnostic probe 的实测结果决定。

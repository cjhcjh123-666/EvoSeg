# 跨帧过程相关工作核对矩阵

核对日期：2026-09-21。只记录作者论文、项目页和官方代码仓库中能够直接核对的内容；“待验证”不是论文结论。

| 工作 | 已公开机制（原文核对） | 与 EvoSeg 候选方向的重合 | 在本项目中仍待验证 |
|---|---|---|---|
| TrajSeg | 双向 text↔trajectory 对齐：既训练 text-to-trajectory，也训练 trajectory-to-text；用 frame-level content integration 将轨迹级 token 适配到逐帧信息，再由统一 mask decoder 输出全帧掩码。官方仓库给出两阶段 image→video 训练和 uniform-frame 推理代码。 | 显式对象轨迹/过程表示；语言条件不只作为一次性的静态 `[SEG]` token。 | 同对象 Static/Dynamic 差异是否已能由原始 Sa2VA 表示；轨迹监督的增益是否超出单纯增加可见帧；当前公开仓库与论文完整配置/权重的可复现性。 |
| EVIS | text-guided Event Queries 将复合视频分解为简单事件；EAFM 用 event-intra 与 event-inter attention 建模事件内、事件间关系；Object-Pixel-Hybrid Learning 联合对象 query 与像素特征。 | 直接对应“对象经历的跨帧过程/事件”表示，也与对象级、像素级联合建模重合。 | 事件划分是否在 Long-RVOS 的原始 dynamic 描述上带来对象内配对收益；收益来自事件结构还是更多时序 token；截至核对时未找到作者公开代码仓库，论文仅承诺发布代码/模型。 |
| VIRST | Spatio-Temporal Fusion 将 segmentation-aware video features 注入 VLM；Temporal Dynamic Anchor Updater 维护时间相邻的动态锚帧，用于大运动、遮挡和重现。官方仓库已有训练、评测代码与 checkpoint 链接。 | 与“VLM 表示和像素分割接口需显式连接”及动态提示/锚点更新重合。 | 在当前固定提示帧的 Sa2VA/SAM2 接口中，瓶颈首先来自语言表征还是固定锚点；需先完成本轮 VLM 帧数控制，不能由 VIRST 结果替代接口诊断。 |
| STAC | 先用 state-space recurrence 做双向空间、因果时间聚合，再做层级的时间→空间自适应压缩；压缩决策用分割目标端到端训练。项目页报告约 85% token 减少和 1.8× 加速，并提供官方代码仓库。 | 与长视频时序上下文和 token 预算控制重合，尤其适合本轮若观察到“更多帧有效但成本过高”的后续问题。 | 本轮 B 允许总 token 随 N 增加，不能用其结果声称固定预算；只有确认 N=32 相对 N=8 的准确率收益后，才值得验证压缩是否保留该收益。 |

## 官方来源

- TrajSeg paper: https://arxiv.org/abs/2603.21488
- TrajSeg code: https://github.com/haodi19/TrajSeg
- EVIS paper: https://arxiv.org/abs/2606.26994
- VIRST paper/project: https://arxiv.org/abs/2603.27060 and https://aidaslab.github.io/VIRST/
- VIRST code: https://github.com/AIDASLab/VIRST
- STAC paper/project: https://arxiv.org/abs/2607.02922 and https://ashesham.github.io/projects/stac/
- STAC code: https://github.com/MCG-NKU/nku-video/tree/main/stac


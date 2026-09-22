# Temporal-process 相关机制核对

核对日期：2026-09-22（Asia/Shanghai）。本表只依据论文原文、项目主页和作者官方仓库；它不替代本项目的 Long-RVOS 实测，也不用于提前选择 Method Route。

| 工作 | 已有机制（原文/代码） | 与候选方法的重合 | 本项目仍需实测的问题 | 官方实现状态 |
|---|---|---|---|---|
| [TrajSeg](https://arxiv.org/abs/2603.21488) | 双向 text–trajectory alignment；用 frame-level content integration 将 trajectory-level token 适配为逐帧信息，再由统一 mask decoder 输出全帧分割。 | 与 Route C 的 multiple / temporally resolved segmentation states 高度重合；也说明“单一汇总 token 到像素解码”不是唯一接口设计。 | 官方 checkpoint 是否可取得；在完全相同 Static/Dynamic 对象配对上是否缩小 gap；收益来自双向轨迹监督、FCI，还是解码器。 | [作者官方仓库](https://github.com/haodi19/TrajSeg)，审计 commit `eaeeb0315b50631ca1c10ff054968ac89940eed5`。训练和推理代码已公开；README 只列基础 LLaVA/SAM2 权重与用户自行训练的两阶段 checkpoint，未给可直接核验的 TrajSeg 成品 checkpoint，因此本轮未把它当成 VIRST 的结果导向替代。 |
| [EVIS](https://arxiv.org/abs/2606.26994) | text-guided learnable Event Queries 将长视频拆成多个相关事件；Object-Pixel-Hybrid Learning 把对象 query 与细粒度像素特征结合以支持长期追踪。 | Event Query 与 Temporal Compiler 的事件/阶段状态有直接重合；object–pixel hybrid interface 与 Route D 的 temporal representation→pixel executor 相近。 | Event Query 是否真的改善同对象 Dynamic–Static gap；事件分解是否需要额外监督；对象 query 是否比 candidate-track selection 更稳。 | 论文原文写明代码和训练模型“will be publicly released”；截至审计时，论文页未链接作者官方实现，作者相关 GitHub 搜索也未找到 EVIS 仓库。因此只作机制核对，不作为可运行基线。 |
| [VIRST](https://arxiv.org/abs/2603.27060) | Spatio-Temporal Fusion 将 segmentation-aware video features 注入 VLM；Temporal Dynamic Anchor Updater 维护时间相邻 anchor，应对大运动、遮挡与重现；端到端统一全局视频推理和像素预测。 | STF 与 Route D 的 grounding interface 直接重合；TDAU 是显式 temporal organization，对本轮“显式 temporal VLM 是否缩小 gap”最关键。 | 在同一 Long-RVOS paired manifest 上的模型内 Dynamic–Static gap；其原生至多 64 VLM 帧/32 SAM prompt 帧协议是否优于 Sa2VA，而不是依赖跨模型绝对 J&F。 | [作者官方仓库](https://github.com/AIDASLab/VIRST)，本轮固定 commit `00aecefdcbb1b5f87f9913a58d2289ce0ab82f66`、官方 checkpoint；已启动真实 64-object protocol pilot。 |
| [STAC](https://arxiv.org/abs/2607.02922) | State-informed Spatiotemporal Aggregator 先以双向空间扫描和因果时间扫描建立上下文；Hierarchical State-adaptive Compression 再先时间、后空间压缩，并用分割目标训练保留决策。官方报告约 85% token reduction 与 1.8× speedup。 | 与 Route B 的 compact temporal compiler / train-heavy deploy-light 最接近；它解决高效保留时序 token，而不是 candidate target selection。 | 在本项目 Static/Dynamic 配对中，压缩是否保持或改善 Dynamic gap；额外效率是否能在同分辨率、同帧范围下复现；不能把论文的跨数据集结果直接当成本项目证据。 | [作者官方仓库](https://github.com/MCG-NKU/nku-video/tree/main/stac)，审计父仓 commit `9cf0733548e43672bc1820046ff5c68aba3d0cca`；代码、环境、训练/推理说明和 checkpoint 下载项已公开。 |

## 对当前决策的约束

- TrajSeg 和 EVIS 已覆盖“多时序状态/事件 query”这一设计空间；若进入 Route C，必须证明新的状态接口与它们的差异，而不能只换命名。
- VIRST 是当前跨模型实验里唯一明确带 STF/TDAU 的显式 temporal VLM；必须等同对象配对结果，不能以论文总体榜单代替 Dynamic gap 检验。
- STAC 的主要贡献是先建立时序上下文再压缩。只有实测表明显式 temporal 模型缩小 gap 且计算代价显著时，Route B 才有依据。
- Candidate Track Matcher 与以上工作的关键差异应是：GT-free candidate bank 先覆盖对象，再由 query-conditioned 时序序列选择 track。该路线是否成立仍取决于 SAM3.1 oracle coverage，而不是文献相似性。

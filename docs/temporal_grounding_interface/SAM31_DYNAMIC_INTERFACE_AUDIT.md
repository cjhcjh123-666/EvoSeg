# SAM 3.1 动态接口审计

## 审计对象

- 官方仓库：`facebookresearch/sam3`，本地 `main` commit `2345a4ad109ac29c569da749c91d84f10dc08c40`，工作树干净。
- 版本依据：官方 `RELEASE_SAM3p1.md`（2026-03-27），Object Multiplex checkpoint。
- checkpoint：`sam3.1_multiplex.pt`，SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- 隔离环境：Python 3.12，PyTorch `2.7.1+cu118`，A800 80GB；未修改共享 shell/CUDA 环境。
- 原则：不修改 Meta checkout，不调用私有核心逻辑来制造实验效果。

## 官方公开 wrapper 能力

`sam3/model/sam3_base_predictor.py` 的 `Sam3BasePredictor` 是本实验采用的公开 session wrapper。

| 操作 | 公开入口 | SAM 3.1 Multiplex 中途使用 | 审计结论 |
|---|---|---:|---|
| text prompt | `add_prompt(text=..., frame_idx=...)` | 不作为连续 correction | Multiplex 的 semantic prompt 路径会重置 state；可开始一次 detection/tracking，但不能宣称维持同一 identity 的中途文本更新。 |
| point prompt | `add_prompt(points=..., point_labels=..., obj_id=...)` | 是 | 指定已有 `obj_id` 时进入官方 refine 路径，更新同一 tracker object；本实验唯一采用的动态 correction API。 |
| box prompt | `add_prompt(bounding_boxes=...)` | 不作为连续 correction | 与 semantic prompt 共用重置路径；适合初始化/新 session，不适合本实验的 identity-preserving update。 |
| mask prompt | `add_mask(...)` | 否 | wrapper 明确检查 `add_tracker_new_mask`；Multiplex 不提供该能力并抛出 `NotImplementedError`。 |
| remove object | `remove_object(...)` | 是 | 正式公开入口，但本实验不通过删除/重建对象规避 identity 维护。 |
| propagation | `propagate_in_video(direction, start, max_frames)` | 是 | 支持 forward/backward/both 和有界传播。 |
| session memory | `start_session/reset_session/close_session` | 是 | point refinement 通过同一 session、同一 `obj_id` 使用 tracker memory。 |

## 内部实现（不作为稳定 API）

`sam3_multiplex_tracking.py` 中 tracker 的 `add_new_points` / `add_new_mask`、`add_tracker_new_points` 等属于模型内部实现。它们解释了公开 point refinement 如何写入 conditioning frame 和 memory，但本实验不会直接调用这些内部函数。尤其是内部 tracker 存在 mask 写入，不代表 Multiplex 的公开 wrapper 支持 `add_mask`。

## 本轮采用的 Normal Route

官方 point correction 足以实现 identity-preserving 动态更新，因此不需要把主实验降级为“独立 chunk 重跑 + identity stitching”proxy：

1. 当前 stage 用官方 text/concept prompt 在独立候选 session 的当前 anchor frame 产生 candidate objects；候选生成函数不接收 GT。
2. 冻结 Sa2VA 从累计可见帧或严格匹配的 anchor-only control 得到 `z_k`，并在 anchor frame 生成 grounding mask。
3. 以 grounding mask 与候选 mask 的 IoU 选择候选；这是零训练参数、两条件完全相同的 scorer。GT 不参与选择。
4. 从被选 candidate mask 内部确定性取一个正点，通过公开 `add_prompt(points=..., point_labels=[1], obj_id=<固定ID>)` 更新维护中的 SAM 3.1 track。
5. 所有 stage 保持同一 session 和同一 tracker `obj_id`，再调用公开 propagation。

运行时生命周期还有两个必须显式处理的边界：首个有候选 stage
出现之前不能调用 propagation（否则 Multiplex 会进入没有 tracker backbone
state 的 full-VG 路径）；首个有效 point 建立 track 后，每次 correction 都按
官方公开 API 连续执行当前阶段的 backward propagation 和到视频末端的
forward propagation。后者为下一阶段提供 pre-update hypothesis。没有候选的
阶段保持空 mask，不访问或伪造私有 tracker state。

Static control 重复当前 anchor 图像到与 temporal branch 相同的图像/token槽位，不暴露其他帧；Temporal branch 使用从视频开始至当前 anchor 的均匀累计采样。两者使用相同 checkpoint、decoder、候选、update 次数和 SAM 3.1 API。

## 受控边界

- 不把中途 text/box 调用描述为 continuous update；它会重置 semantic state。
- 不直接调用内部 mask/point tracker 函数。
- 候选点只由 candidate mask 计算；GT 只在完整推理结束后做指标与 correction transition 诊断。
- 真实等价性 smoke 表明，把每次 correction 的 forward propagation 强行截断到
  下一 stage 会改变 K=4/K=8 的最终 mask 和指标，尽管 candidate selection 相同；
  因而该优化被撤回，正式结果使用上节所述的完整公开 propagation 生命周期。
- 仅一次、在最后 stage 初始化再全片反向传播的 Single-global 条件出现
  275/275 expression 全空输出；一个独立复现样本即使 grounding mask 与所选
  candidate 的 IoU=0.934，公开 point 初始化的输出面积仍为 0。该条件将在最终
  gate 中标记为 protocol-invalid，不会利用空 GT 帧对空预测给出的 J/F=1 宣称
  有效基线或 gap 改善。

# Final GO / NO-GO

当前决策：**NO-GO**。

## Gate

- A：部分满足。K4 Dynamic Temporal−Static = +0.34pp，但 95% CI
  [-1.49, +2.48]pp；K=2/4/8 三档中两档为正。
- B：失败。Temporal K4 Dynamic−Static gap 为 +0.59pp，没有相对有效
  Single-global 基线证明至少 1pp 缩小；实际 Single-global 还是全空输出，协议无效。
- C：失败。Temporal K4 correction rate 只比 Static 高 0.64pp，且被纠正样本的
  Temporal−Static J&F 平均 -1.08pp。

## 结论

当前 frozen Sa2VA stage states + candidate-mask IoU scorer + SAM3.1 public point
refinement 的 Dynamic Grounding Interface，没有证明 temporal-update 优于参数匹配的
static-update。K8 的事后数值较高（+1.37pp，95% CI [-0.03, +3.00]pp），但 CI
仍跨 0，且预注册主比较是 K4，不能据此改选 K 或改写结论。

因此不扩展 274-object full run，不训练 single-forward temporal state compiler，
也不继续堆 Transformer/Agent/RL。这个 NO-GO 只针对本轮具体接口和零训练参数
scorer；Single-global 全空现象同时说明下一步若继续研究，应先修正/验证 SAM3.1
初始化与像素执行协议，而不是把当前负结果外推为所有动态 grounding interface
都无效。

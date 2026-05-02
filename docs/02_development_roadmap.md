# 迭代开发路线图

## 总览

| 阶段 | 名称 | 核心目标 | 是否必须 |
| --- | --- | --- | --- |
| 00 | 读懂原 MinivLLM | 搞清楚 CUDA 版执行链路 | 必须 |
| 01 | 最小 PyTorch 推理闭环 | 跑通一个 tiny model 生成循环 | 必须 |
| 02 | NPU eager 后端 | 学会 `torch_npu` 基本接入 | 必须 |
| 03 | Qwen3 结构与权重 | 跑真实小模型 | 必须 |
| 04 | 连续 KV cache | 理解 decode 加速 | 必须 |
| 05 | paged KV cache | 理解 vLLM 核心数据结构 | 必须 |
| 06 | Ascend attention 算子 | 替换 PyTorch attention | 必须 |
| 07 | batch engine | 接近 mini-vLLM 形态 | 必须 |
| 08 | graph 与性能实验 | 学习 NPU graph | 可选 |
| 09 | tensor parallel | 学习多卡 HCCL/TP | 可选 |

## 为什么这样排序

新人最容易卡在两个地方：

- 一开始就碰高性能 attention kernel。
- 一开始就碰多卡和 graph。

这两个点都会让问题混在一起：模型结构、shape、设备、算子、通信、内存、编译全都同时出错。正确的学习路线是先让每一层都可运行，再逐层替换。

## 每个阶段的推进方式

每个阶段按这个顺序做：

1. 读阶段文档。
2. 写最小代码。
3. 跑一个最小脚本。
4. 写一个最小测试。
5. 补阶段笔记。
6. 提交一个清晰 commit。

推荐 commit 形态：

```text
docs: add original minivllm architecture notes
feat: add tiny transformer inference loop
feat: add backend abstraction for cpu and npu
feat: add contiguous kv cache
feat: add paged kv cache torch implementation
```

## 什么时候可以进入下一阶段

只有当前阶段满足验收标准，才进入下一阶段。

不要因为“代码看起来差不多”就跳过验证。推理引擎很多 bug 都是 shape 正确但语义错误，例如：

- position 算错。
- KV cache 写入错位。
- block table 指向错 block。
- decode 时 context length 少加一。
- prefill 只应该取最后 token logits，但错误取了所有 logits。

## 建议里程碑

### M1：跑起来

覆盖阶段 00 到 02。

结果：tiny model 能在 CPU 和 NPU eager 跑。

### M2：真实模型

覆盖阶段 03 到 04。

结果：Qwen3-0.6B 能加载权重，并用连续 KV cache decode。

### M3：mini-vLLM 核心

覆盖阶段 05 到 07。

结果：paged KV cache、scheduler、batch engine 可运行。

### M4：Ascend 专项优化

覆盖阶段 08 到 09。

结果：尝试 graph、多卡、性能分析。

## 推荐文档习惯

每个阶段结束后，在阶段笔记里记录：

- 关键 shape。
- 关键函数调用链。
- 当前实现的限制。
- 和 vLLM 或原 MinivLLM 的差异。
- 遇到的报错和解决方式。

这样你最后不只是有代码，还有一份自己的 AI infra 学习档案。

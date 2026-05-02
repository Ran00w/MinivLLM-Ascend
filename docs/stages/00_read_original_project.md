# Stage 00：读懂原 MinivLLM

## 学习目标

搞清楚原 CUDA 版 MinivLLM 的执行链路，知道哪些模块可以复用，哪些模块必须重写。

## 本阶段不做什么

- 不写 NPU 代码。
- 不改模型结构。
- 不写新 kernel。
- 不追性能。

## 需要阅读的原项目文件

```text
src/myvllm/engine/sequence.py
src/myvllm/engine/scheduler.py
src/myvllm/engine/block_manager.py
src/myvllm/engine/model_runner.py
src/myvllm/layers/attention.py
src/myvllm/models/qwen3.py
src/myvllm/layers/embedding_head.py
```

## 要产出的文件

```text
docs/notes/original_minivllm_arch.md
docs/notes/original_attention_path.md
docs/notes/cuda_to_npu_gap.md
```

## 任务清单

- 画出从 prompt 到 token 输出的调用链。
- 解释 `Sequence` 保存了什么状态。
- 解释 scheduler 为什么区分 prefill 和 decode。
- 解释 block manager 如何分配 block。
- 解释 block table 是什么。
- 找出所有 CUDA-only 代码。
- 找出所有 Triton kernel。
- 找出哪些地方用了 `torch.compile`。
- 总结哪些模块可以直接搬，哪些模块要改。

## 重点问题

你应该能回答：

- prefill 阶段输入是什么 shape？
- decode 阶段为什么每个 sequence 只输入一个 token？
- KV cache 是什么时候写入的？
- sampler 为什么只需要最后一个 token 的 logits？
- prefix cache 为什么只能复用完整 block？
- CUDA Graph 在原项目里优化了什么？

## 验收标准

写出一篇架构笔记，至少包含：

- 一张模块关系图。
- 一条完整调用链。
- 一个 CUDA/NPU 差异表。
- 一个你自己的阶段总结。

## 阶段完成后的下一步

进入 Stage 01，用最简单的 PyTorch tiny model 重新写一个推理闭环。目的不是复用所有代码，而是先确认自己理解了推理循环。

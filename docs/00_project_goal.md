# 项目总目标

## 一句话目标

从零到一复刻一个用于学习的 mini-vLLM Ascend/NPU 版本，帮助理解 LLM 推理引擎、KV cache、paged attention、调度器和 Ascend 后端。

## 为什么不直接改原 MinivLLM

原项目是 CUDA/GPU 思路：

- device 写死为 CUDA。
- attention 依赖 Triton kernel。
- graph 依赖 CUDA Graph。
- 分布式后端使用 NCCL。
- 显存统计使用 `torch.cuda` API。

Ascend 不是 CUDA 兼容设备。学习项目应该把这些硬件相关能力抽象出来，然后逐步替换为 NPU 能力。

## 目标用户

这个项目默认读者是 AI infra 新人，已经会一点 Python/PyTorch，但还没有完整写过推理引擎。

文档要避免只写结论，要解释：

- 这个模块为什么存在。
- 输入输出是什么。
- 它和上一个阶段有什么关系。
- 当前实现为什么先简单、后优化。

## 最终可接受形态

最终项目不要求对齐 vLLM 生产级性能，但应该能清楚展示这些机制：

- tokenizer 把 prompt 转成 token ids。
- `Sequence` 表示一个正在生成的请求。
- `Scheduler` 决定本轮跑 prefill 还是 decode。
- `ModelRunner` 负责把输入搬到设备并执行模型。
- attention 在 prefill 阶段处理整段 prompt。
- attention 在 decode 阶段只处理新 token，并读取历史 KV cache。
- KV cache 从连续 cache 迭代到 paged cache。
- paged KV cache 通过 block table 找历史 token。
- NPU 后端通过 `torch_npu` 和 Ascend attention 算子接入。

## 非目标

第一阶段不要追求这些内容：

- 生产级吞吐。
- 高并发服务。
- OpenAI API Server。
- 多机多卡。
- 自写 Ascend C kernel。
- 自写 Triton-Ascend kernel。
- 完整兼容 vLLM 所有模型。
- FP8、量化、speculative decoding。

这些内容可以作为长期扩展，但不应阻塞主线学习。

## 推荐最终能力

最低目标：

- 单卡 NPU。
- eager 模式。
- 一个小模型。
- 单 prompt 推理。
- 简单 KV cache。

进阶目标：

- 多 prompt batch。
- paged KV cache。
- prefix cache。
- 使用 `torch_npu` attention 算子。
- graph replay。
- 单机多卡 tensor parallel。

## 每个阶段都要产出的东西

每个阶段结束时至少要有：

- 可以运行的代码，哪怕很慢。
- 一个最小 demo。
- 一个测试或手工验证脚本。
- 一篇阶段学习笔记。
- 一段“下一阶段为什么需要做”的说明。

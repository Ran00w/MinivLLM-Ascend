# Stage 08：graph 与性能实验

## 学习目标

学习 Ascend 上的 graph capture 和基础性能分析。

这一阶段是优化阶段，不是功能阶段。必须在 eager 版本已经稳定后再做。

## 本阶段不做什么

- 不改变模型语义。
- 不重写 scheduler。
- 不引入多卡。
- 不把 graph 作为唯一执行路径。

## 建议新增或修改文件

```text
src/minivllm_ascend/backend/
  graph_npu.py
src/minivllm_ascend/engine/
  model_runner.py
scripts/
  benchmark_prefill.py
  benchmark_decode.py
  run_decode_graph.py
docs/notes/
  npu_graph_notes.md
```

## 任务清单

- 先 benchmark eager decode。
- 固定 batch size。
- 固定 max block table shape。
- 预分配 input/output tensor。
- 尝试 `torch.npu.NPUGraph` 或当前 torch_npu 推荐 graph API。
- 对比 eager 和 graph replay。
- 记录哪些 op 不能 capture。
- 记录 graph 对 shape 静态性的要求。

## 为什么 decode 更适合先做 graph

decode 每一步输入 shape 更稳定：

```text
input_ids: [batch]
slot_mapping: [batch]
context_lens: [batch]
block_tables: [batch, max_num_blocks]
```

prefill 的总 token 数变化更大，不适合早期 graph 实验。

## 验收标准

- 固定 batch size 的 decode step 能 graph replay。
- eager 和 graph 输出 shape 一致。
- benchmark 能打印 tokens/sec 或 step latency。

## 常见坑

- graph capture 时有动态 shape。
- capture 后传入了不同 shape。
- capture 中包含 CPU 同步。
- capture 前没有 warmup。
- graph 输入 tensor 被重新分配，而不是原地 copy。

## 阶段完成后的下一步

进入 Stage 09，学习多卡 tensor parallel。如果只做单卡学习项目，本阶段之后可以转向整理文档和写博客。

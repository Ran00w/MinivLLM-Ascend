# Stage 09：多卡 tensor parallel

## 学习目标

学习单机多 NPU 上的 tensor parallel。

这一阶段会引入 HCCL、进程组、权重切分、通信同步，是最复杂的扩展阶段。建议最后再做。

## 本阶段不做什么

- 不做 pipeline parallel。
- 不做 expert parallel。
- 不做多机。
- 不追求和 vLLM 相同性能。

## 建议新增或修改文件

```text
src/minivllm_ascend/backend/
  distributed_npu.py
src/minivllm_ascend/layers/
  linear.py
  embedding_head.py
src/minivllm_ascend/engine/
  model_runner.py
  llm_engine.py
scripts/
  run_qwen3_tp.py
tests/
  test_parallel_linear.py
```

## 任务清单

- 学习 HCCL 进程组初始化。
- 实现 tensor parallel rank/world_size 获取。
- 实现 column parallel linear。
- 实现 row parallel linear。
- 实现 vocab parallel embedding。
- 实现 parallel lm head。
- 加载权重时按 rank 切分。
- row parallel 输出后 all-reduce。
- lm head logits 按需 gather。
- 单卡和多卡结果 shape 对齐。

## 推荐最小实验

先不要用 Qwen3。先写一个小 linear：

```text
full_linear(x)
```

对比：

```text
column_parallel + row_parallel
```

确认结果接近后，再接模型。

## 关键概念

### Column Parallel

把输出维度切开。每张卡算一部分输出。

### Row Parallel

把输入维度切开。每张卡算一部分结果，然后 all-reduce 求和。

### Vocab Parallel

把词表维度切开。每张卡保存一部分 embedding 或 lm_head。

## 验收标准

- `world_size=1` 和 `world_size=2` 都能启动。
- parallel linear 的 shape 正确。
- Qwen3 小 batch forward 能跑。
- decode 能生成 token。

## 常见坑

- 环境变量没有设置。
- rank 和 device id 不一致。
- HCCL 初始化失败。
- 权重切分维度错。
- all-reduce 放错位置。
- rank 0 才有完整 logits，但其他 rank 还在继续执行 sampler。

## 阶段完成后的下一步

整理整个项目，写最终总结：

```text
docs/final_report.md
```

建议总结：

- 和原 MinivLLM 的差异。
- 和 vLLM Ascend 的差异。
- 哪些地方是学习实现。
- 哪些地方可以继续优化。

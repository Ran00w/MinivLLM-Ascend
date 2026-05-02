# Stage 07：batch engine 和 scheduler

## 学习目标

把前面阶段的能力整合成接近 mini-vLLM 的推理引擎。

这一阶段要支持多个 prompt 一起进入系统，由 scheduler 决定每一步执行哪些 sequence。

## 本阶段不做什么

- 不做 OpenAI API server。
- 不做流式输出服务。
- 不做多机。
- 不追生产级调度策略。

## 建议新增或修改文件

```text
src/minivllm_ascend/engine/
  llm_engine.py
  model_runner.py
  scheduler.py
  sequence.py
  block_manager.py
src/minivllm_ascend/
  sampling_params.py
scripts/
  run_batch_generation.py
tests/
  test_scheduler.py
  test_engine_batch.py
```

## 任务清单

- 实现 `LLMEngine.add_prompt`。
- 实现 `LLMEngine.generate`。
- scheduler 支持 waiting queue。
- scheduler 支持 running queue。
- scheduler 优先调度 prefill。
- 没有 prefill 时调度 decode。
- 每个 sequence 独立判断 EOS 和 max_tokens。
- 支持 `max_num_sequences`。
- 支持 `max_num_batched_tokens`。
- 支持 block 不够时暂停或 preempt。
- 返回每个 prompt 对应的生成结果。

## 推荐最小调度策略

先用简单策略：

1. 如果 waiting queue 里有请求，并且 token/block 资源够，就调度 prefill。
2. 否则调度 running queue 里的 decode。
3. decode 每个 sequence 每步只生成一个 token。
4. sequence 结束后释放 block。

不要一开始做复杂优先级。

## 验收标准

运行：

```bash
python scripts/run_batch_generation.py
```

输入多个 prompt，应该看到：

- prefill 阶段处理多个 prompt。
- decode 阶段每轮处理多个 sequence。
- 不同 sequence 可以在不同时间结束。
- 结束后释放对应 block。

## 建议日志

每一步打印简短日志：

```text
step=0 phase=prefill batch=3 tokens=128 blocks_used=12
step=1 phase=decode batch=3 tokens=3 blocks_used=12
step=2 phase=decode batch=2 tokens=2 blocks_used=10
```

## 常见坑

- prefill 后没有把 sequence 放入 running queue。
- decode 后忘记 append token。
- sequence finished 后忘记释放 block。
- tokenizer decode 顺序和 seq_id 对不上。
- batch 中每个 sequence 长度不同，position 算错。

## 阶段完成后的下一步

进入 Stage 08，研究 graph 和性能。但如果主线学习已经达成，也可以先暂停，整理项目文档和总结。

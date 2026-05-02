# 阶段任务看板

这个文件用于跟踪整个学习项目的推进。每完成一个阶段，建议在对应阶段文档旁边补一篇自己的学习记录。

## 总进度

| 阶段 | 目标 | 主要代码文件 | 脚本 | 测试 | 文档产出 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | 读懂原 MinivLLM | 无 | 无 | 无 | `docs/notes/original_minivllm_arch.md` | 未开始 |
| 01 | 最小 PyTorch 推理闭环 | `tiny_transformer.py`, `simple_engine.py`, `attention_torch.py` | `scripts/run_tiny.py` | `test_sequence.py`, `test_tiny_generation.py` | stage 01 学习记录 | 未开始 |
| 02 | NPU eager 后端 | `backend/base.py`, `backend/torch_cpu.py`, `backend/torch_npu.py` | `scripts/run_tiny.py --device npu` | `test_backend.py` | stage 02 学习记录 | 未开始 |
| 03 | Qwen3 模型与权重 | `models/qwen3.py`, `utils/loader.py`, `rotary_embedding.py` | `scripts/run_qwen3_torch_attention.py` | `test_qwen3_shapes.py`, `test_weight_loading.py` | stage 03 学习记录 | 未开始 |
| 04 | 连续 KV cache | `kv_cache.py`, `attention_kv_torch.py`, `model_runner.py` | `scripts/run_qwen3_kv_cache.py` | `test_kv_cache.py`, `test_decode_step.py` | stage 04 学习记录 | 未开始 |
| 05 | paged KV cache | `block_manager.py`, `paged_attention_torch.py` | `scripts/run_paged_kv_demo.py` | `test_block_manager.py`, `test_paged_attention_torch.py` | stage 05 学习记录 | 未开始 |
| 06 | Ascend attention 算子 | `attention_npu.py`, `attention_backend.py` | `scripts/run_npu_prefill_attention.py`, `scripts/run_npu_decode_attention.py` | `test_attention_shapes.py` | `docs/notes/ascend_attention_shapes.md` | 未开始 |
| 07 | batch engine | `llm_engine.py`, `scheduler.py`, `model_runner.py` | `scripts/run_batch_generation.py` | `test_scheduler.py`, `test_engine_batch.py` | stage 07 学习记录 | 未开始 |
| 08 | graph 和性能 | `backend/graph_npu.py` | `scripts/benchmark_decode.py`, `scripts/run_decode_graph.py` | 可选 | `docs/notes/npu_graph_notes.md` | 可选 |
| 09 | tensor parallel | `distributed_npu.py`, `linear.py`, `embedding_head.py` | `scripts/run_qwen3_tp.py` | `test_parallel_linear.py` | stage 09 学习记录 | 可选 |

## 每阶段固定完成项

每个阶段至少完成这些事情：

- 写或修改阶段要求的代码文件。
- 写一个可以直接运行的脚本。
- 写一个最小测试。
- 运行脚本并记录结果。
- 补一篇学习记录。
- 提交一个独立 commit。

## 推荐 issue 拆分

如果后续用 GitHub Issues 管理，可以按这个粒度开 issue：

```text
docs: write original MinivLLM architecture notes
feat: add tiny transformer inference loop
feat: add backend abstraction for CPU and NPU
feat: add Qwen3 model and safetensors loader
feat: add contiguous KV cache
feat: add paged KV cache torch implementation
feat: add NPU prompt attention wrapper
feat: add NPU incremental attention wrapper
feat: add batch scheduler and LLM engine
perf: experiment with NPU graph decode replay
feat: experiment with HCCL tensor parallel
```

## 阶段完成定义

一个阶段不是“代码写完”就算完成。完成定义是：

```text
代码能跑 + 有脚本 + 有测试或手工验证 + 有文档记录 + 有清晰 commit
```

如果某阶段遇到硬件或环境阻塞，也要写清楚：

- 阻塞命令是什么。
- 报错是什么。
- 当前推测原因是什么。
- 下一步准备如何验证。

# 项目结构与模块边界

## 推荐目录结构

```text
MinivLLM-Ascend/
  README.md
  pyproject.toml
  docs/
    00_project_goal.md
    01_architecture_map.md
    02_development_roadmap.md
    03_git_github_workflow.md
    stages/
    templates/
  scripts/
    run_tiny.py
    run_qwen3.py
    benchmark_prefill.py
    benchmark_decode.py
  tests/
    test_sequence.py
    test_scheduler.py
    test_kv_cache.py
    test_attention_shapes.py
  src/
    minivllm_ascend/
      __init__.py
      sampling_params.py
      backend/
        __init__.py
        base.py
        torch_cpu.py
        torch_npu.py
        graph_npu.py
        distributed_npu.py
      engine/
        __init__.py
        sequence.py
        scheduler.py
        block_manager.py
        kv_cache.py
        model_runner.py
        llm_engine.py
      layers/
        __init__.py
        activation.py
        attention_torch.py
        attention_kv_torch.py
        attention_npu.py
        embedding_head.py
        layernorm.py
        linear.py
        paged_attention_torch.py
        rotary_embedding.py
        sampler.py
      models/
        __init__.py
        tiny_transformer.py
        qwen3.py
        llama.py
      utils/
        __init__.py
        context.py
        loader.py
        logging.py
```

## 核心执行链路

```text
prompt
  -> tokenizer
  -> Sequence
  -> Scheduler
  -> ModelRunner
  -> Model
  -> Attention
  -> KV cache
  -> Sampler
  -> new token
  -> Sequence.append_token
```

## 模块职责

### `backend/`

负责隐藏硬件差异。早期只需要支持 CPU 和 NPU。

典型职责：

- 返回当前设备名。
- 创建 tensor。
- 把 tensor 搬到设备。
- 同步设备。
- 清理缓存。
- 查询内存。
- 可选：graph capture。
- 可选：分布式通信。

不要让业务代码到处出现 `.cuda()`、`.npu()`、`torch.cuda`、`torch.npu`。这些都应该集中在 backend。

### `engine/sequence.py`

表示一个请求。

它应该记录：

- `seq_id`
- prompt token ids
- 已生成 token ids
- 当前状态：waiting、running、finished
- 采样参数
- block table
- 已缓存 token 数

### `engine/scheduler.py`

决定每一步执行哪些 sequence。

它应该回答两个问题：

- 这一步是 prefill 还是 decode？
- 哪些 sequence 会进入本轮 batch？

早期 scheduler 可以非常简单。后期再加入 block 数限制、preempt、prefix cache。

### `engine/block_manager.py`

管理 paged KV cache 的逻辑块。

它不直接存 tensor。它只管：

- 哪些 block 空闲。
- 哪些 block 被 sequence 占用。
- sequence 的 block table。
- prefix cache hash。
- block 引用计数。

### `engine/kv_cache.py`

存真实 KV tensor。

前期可以是连续 KV cache：

```text
k_cache: [batch, max_seq_len, num_kv_heads, head_dim]
v_cache: [batch, max_seq_len, num_kv_heads, head_dim]
```

后期改为 paged KV cache：

```text
k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

### `engine/model_runner.py`

负责一次模型执行。

典型职责：

- 根据 sequence 准备 input ids。
- 准备 positions。
- 准备 slot mapping。
- 准备 block tables。
- 调用模型 forward。
- 调用 sampler。
- 返回新 token。

### `layers/attention_torch.py`

最朴素的 PyTorch attention。用于理解正确性。

### `layers/attention_kv_torch.py`

带连续 KV cache 的 attention。用于理解 decode 为什么只需要输入最后一个 token。

### `layers/paged_attention_torch.py`

用 PyTorch gather 实现 paged attention。性能不重要，重点是理解 block table。

### `layers/attention_npu.py`

封装 Ascend attention 算子。尽量把 `torch_npu` 的复杂 shape/layout 处理集中在这里。

### `models/`

模型结构。

推荐顺序：

1. `tiny_transformer.py`
2. `qwen3.py`
3. `llama.py`

### `scripts/`

只放可以直接运行的 demo 或 benchmark。

脚本越小越好。一个脚本只验证一件事。

### `tests/`

测试不要求覆盖所有数值细节，但要覆盖形状、状态变化和关键数据结构。

优先测试：

- `Sequence.append_token`
- scheduler prefill/decode 切换
- block table 分配
- KV cache 写入位置
- attention 输入输出 shape

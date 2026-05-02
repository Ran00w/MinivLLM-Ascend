# Stage 01：最小 PyTorch 推理闭环

## 学习目标

不用真实大模型，先写一个 tiny Transformer，让它能完成最小生成循环。

这一阶段的重点是理解：

- token ids 如何输入模型。
- logits 如何变成下一个 token。
- 生成循环如何停止。
- prefill 和 decode 的概念雏形。

## 本阶段不做什么

- 不接 NPU。
- 不加载 Qwen3 权重。
- 不做 KV cache。
- 不做 paged attention。
- 不做 batch scheduler。

## 建议新增文件

```text
src/minivllm_ascend/
  __init__.py
  sampling_params.py
  models/
    __init__.py
    tiny_transformer.py
  layers/
    __init__.py
    attention_torch.py
    layernorm.py
    sampler.py
  engine/
    __init__.py
    sequence.py
    simple_engine.py
scripts/
  run_tiny.py
tests/
  test_sequence.py
  test_tiny_generation.py
```

## 模块说明

### `models/tiny_transformer.py`

实现一个很小的 Transformer decoder-only 模型。

它可以随机初始化，不需要输出有意义文本。

### `layers/attention_torch.py`

使用普通 PyTorch 写 causal attention。

这一版可以直接构造完整 attention matrix。慢没关系。

### `layers/sampler.py`

先支持 greedy sampling：

```text
next_token = logits.argmax(dim=-1)
```

后续再加 temperature sampling。

### `engine/simple_engine.py`

实现最简单的循环：

```text
input_ids -> model -> logits -> sampler -> append token -> repeat
```

## 任务清单

- 实现 `SamplingParams`。
- 实现 `Sequence`。
- 实现 tiny embedding、attention、MLP、lm_head。
- 实现 causal mask。
- 实现 greedy sampler。
- 实现 `SimpleEngine.generate`。
- 写 `scripts/run_tiny.py`。
- 写测试验证 `Sequence.append_token`。
- 写测试验证生成长度不会超过 `max_tokens`。

## 验收标准

运行：

```bash
python scripts/run_tiny.py
```

应该看到：

- 输入 token ids。
- 每一步生成的新 token。
- 最终 token ids。

输出不需要像自然语言，只要流程正确。

## 学习记录建议

记录这些 shape：

```text
input_ids: [batch, seq_len]
hidden_states: [batch, seq_len, hidden_size]
q/k/v: [batch, num_heads, seq_len, head_dim]
logits: [batch, seq_len, vocab_size]
next_token: [batch]
```

## 阶段完成后的下一步

进入 Stage 02，把 tiny 推理闭环迁移到 NPU eager。重点是设备抽象，而不是模型能力。

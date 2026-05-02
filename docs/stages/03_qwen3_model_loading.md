# Stage 03：Qwen3 模型结构与权重加载

## 学习目标

把 tiny model 换成真实的小模型结构，例如 Qwen3-0.6B。

这一阶段重点是模型结构和权重，不是高性能 attention。

## 本阶段不做什么

- 不做 paged attention。
- 不做 Ascend fused attention。
- 不做 graph。
- 不做 tensor parallel。
- 不追求输出速度。

## 建议新增或修改文件

```text
src/minivllm_ascend/models/
  qwen3.py
src/minivllm_ascend/layers/
  activation.py
  embedding_head.py
  layernorm.py
  linear.py
  rotary_embedding.py
  attention_torch.py
src/minivllm_ascend/utils/
  loader.py
  context.py
scripts/
  run_qwen3_torch_attention.py
tests/
  test_qwen3_shapes.py
  test_weight_loading.py
```

## 任务清单

- 实现 RMSNorm。
- 实现 SiLU and Mul。
- 实现 RoPE。
- 实现 QKV projection。
- 实现 MLP。
- 实现 LM head。
- 写 HF safetensors 权重加载器。
- 支持 Qwen3 的 q/k/v 权重合并。
- 支持 gate/up 权重合并。
- 使用普通 PyTorch attention 跑通 forward。
- prefill 后只取每个 sequence 最后 token 的 logits。

## 推荐先固定的参数

先只支持一个模型，例如：

```text
Qwen3-0.6B
```

不要一开始写通用模型注册系统。先跑通一个，再抽象。

## 关键 shape

```text
input_ids: [seq_len]
hidden_states: [seq_len, hidden_size]
q: [seq_len, num_heads, head_dim]
k: [seq_len, num_kv_heads, head_dim]
v: [seq_len, num_kv_heads, head_dim]
attention_output: [seq_len, num_heads * head_dim]
logits: [seq_len, vocab_size]
```

## 验收标准

运行：

```bash
python scripts/run_qwen3_torch_attention.py
```

应该能：

- 加载 tokenizer。
- 加载 safetensors 权重。
- 执行一次 prefill。
- 输出 logits shape。
- 采样一个 token。

如果输出文本不稳定，先不用管。当前阶段只验证模型结构和权重能跑。

## 常见坑

- Qwen3 的 `head_dim` 和 `hidden_size / num_heads` 不一致时配置写错。
- Q/K norm 漏掉。
- RoPE base 写错。
- prefill 多序列时 position 没有按序列重置。
- 权重名映射不完整。

## 阶段完成后的下一步

进入 Stage 04，实现连续 KV cache，让 decode 不再每步重复计算所有历史 token。

# Stage 04：连续 KV cache

## 学习目标

实现最容易理解的 KV cache：连续数组。

这一步要理解 autoregressive decoding 的核心优化：

- prefill 计算 prompt 的 K/V 并保存。
- decode 每一步只输入新 token。
- attention 读取历史 K/V。

## 本阶段不做什么

- 不做 paged KV cache。
- 不做 prefix cache。
- 不做 block manager。
- 不做 NPU fused attention。

## 建议新增或修改文件

```text
src/minivllm_ascend/engine/
  kv_cache.py
  model_runner.py
src/minivllm_ascend/layers/
  attention_kv_torch.py
scripts/
  run_qwen3_kv_cache.py
tests/
  test_kv_cache.py
  test_decode_step.py
```

## 推荐 KV cache layout

```text
k_cache: [batch, max_seq_len, num_layers, num_kv_heads, head_dim]
v_cache: [batch, max_seq_len, num_layers, num_kv_heads, head_dim]
```

也可以每层单独保存：

```text
layer.k_cache: [batch, max_seq_len, num_kv_heads, head_dim]
layer.v_cache: [batch, max_seq_len, num_kv_heads, head_dim]
```

新人阶段推荐每层单独保存，更容易调试。

## 任务清单

- 定义 `KVCache` 类。
- prefill 时把整段 prompt 的 K/V 写入 cache。
- decode 时把新 token 的 K/V 写入当前位置。
- decode attention 读取 `[0:context_len]` 的历史 K/V。
- 实现 `context_lens`。
- 实现 `positions = context_lens - 1`。
- 写单步 decode 测试。
- 验证每一步只向模型输入最后一个 token。

## 关键概念

### prefill

```text
input_ids: [prompt_len]
q/k/v: [prompt_len, heads, head_dim]
写入 cache 位置: 0 到 prompt_len - 1
输出: prompt 最后 token 的 logits
```

### decode

```text
input_ids: [1]
q/k/v: [1, heads, head_dim]
写入 cache 位置: context_len - 1
读取 cache 位置: 0 到 context_len - 1
输出: 当前 token 的 logits
```

## 验收标准

- prompt 长度为 N 时，prefill 后 cache 写入 N 个位置。
- decode 第一步只输入 1 个 token。
- decode 后 `context_len` 增加 1。
- 能连续生成多个 token。

## 常见坑

- decode 时先 append token 还是先写 cache，顺序搞混。
- position 使用了 0，而不是当前 token 的真实位置。
- prefill 输出取了所有 logits，而不是最后 token logits。
- KV cache dtype 和模型 dtype 不一致。

## 阶段完成后的下一步

进入 Stage 05，把连续 cache 改成 paged cache，学习 vLLM 的核心内存管理方式。

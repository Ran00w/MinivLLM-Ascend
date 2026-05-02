# Stage 05：paged KV cache 学习版

## 学习目标

实现 vLLM 最核心的思想之一：paged KV cache。

连续 KV cache 的问题是每个请求都要预留最大长度，浪费显存。paged KV cache 把 KV 拆成固定大小 block，由 block table 把逻辑 token 位置映射到物理 block。

## 本阶段不做什么

- 不写 Triton kernel。
- 不写 Ascend C kernel。
- 不接 Ascend fused attention。
- 不追求速度。

## 建议新增或修改文件

```text
src/minivllm_ascend/engine/
  block_manager.py
  sequence.py
  scheduler.py
  kv_cache.py
src/minivllm_ascend/layers/
  paged_attention_torch.py
scripts/
  run_paged_kv_demo.py
tests/
  test_block_manager.py
  test_paged_kv_cache.py
  test_paged_attention_torch.py
```

## 推荐 paged cache layout

```text
k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

每个 sequence 保存：

```text
block_table: [physical_block_id_0, physical_block_id_1, ...]
```

## 任务清单

- 实现 `Block`。
- 实现 `BlockManager`。
- 实现 block 分配和释放。
- 实现 `Sequence.block_table`。
- 实现 `slot_mapping`，表示当前 token 要写入哪个全局 cache slot。
- 实现 PyTorch 版 paged attention。
- 实现简单 prefix cache hash。
- 写测试覆盖 block 分配、释放、复用。
- 写 demo 展示多个 sequence 共享前缀 block。

## block table 示例

假设：

```text
block_size = 4
sequence tokens = [10, 11, 12, 13, 14, 15]
block_table = [7, 3]
```

含义：

```text
逻辑 token 0-3 存在物理 block 7
逻辑 token 4-5 存在物理 block 3
```

查第 5 个逻辑 token：

```text
logical_pos = 5
block_index = logical_pos // block_size = 1
block_offset = logical_pos % block_size = 1
physical_block = block_table[1] = 3
真实位置 = k_cache[3, 1]
```

## PyTorch 版 paged attention

这一版可以很慢：

1. 根据 block table gather 出完整 K/V。
2. 拼成连续 K/V。
3. 调用普通 attention。

慢没关系。目的是验证 block table 语义。

## 验收标准

- sequence 可以分配多个 block。
- decode 新 token 可以写入正确 block offset。
- block 不够时 scheduler 能停止分配或 preempt。
- paged attention 输出 shape 正确。
- prefix 相同的两个 sequence 能复用完整 block。

## 常见坑

- partial block 不应该进入 prefix cache。
- block 引用计数没有增加或减少。
- block table padding 用了有效 block id。
- slot mapping 使用 token index，而不是全局 cache slot。
- block size 变化后测试没覆盖。

## 阶段完成后的下一步

进入 Stage 06，尝试把 PyTorch attention 替换成 Ascend 官方 attention 算子。

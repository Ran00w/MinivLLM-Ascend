# Stage 06：Ascend attention 算子

## 学习目标

把 PyTorch attention 替换为 Ascend NPU fused attention 算子。

这是整个项目最难的必做阶段。难点不只是 API，而是 shape、layout、dtype、mask、GQA、block table 都要对齐。

## 本阶段不做什么

- 不自写 Ascend C kernel。
- 不自写 Triton-Ascend kernel。
- 不做多卡。
- 不做 graph。

## 建议新增或修改文件

```text
src/minivllm_ascend/layers/
  attention_backend.py
  attention_npu.py
  attention_torch.py
  paged_attention_torch.py
scripts/
  inspect_npu_attention_api.py
  run_npu_prefill_attention.py
  run_npu_decode_attention.py
tests/
  test_attention_shapes.py
  test_attention_backend_select.py
docs/notes/
  ascend_attention_shapes.md
```

## 推荐学习顺序

### 1. 先跑 prefill attention

研究 `torch_npu.npu_prompt_flash_attention` 或当前 CANN/torch_npu 版本推荐的 prompt flash attention API。

目标：

- 单 batch。
- 固定 seq_len。
- 不接 KV cache。
- 只验证输出 shape。

### 2. 再跑 decode attention

研究 `torch_npu.npu_incre_flash_attention` 或当前版本推荐的 incremental flash attention API。

目标：

- query 只有一个 token。
- K/V 从已有 cache 读取。
- 不接 paged cache。

### 3. 最后接 paged attention

重点关注：

- `block_table`
- `block_size`
- cache layout
- `num_heads`
- `num_key_value_heads`
- GQA
- attention mask
- actual sequence length

## 任务清单

- 写一个脚本单独打印当前 `torch_npu` 版本。
- 单独构造 q/k/v tensor 调用 prompt attention。
- 单独构造 q/k/v cache 调用 incremental attention。
- 记录 API 需要的 layout。
- 写 shape 转换函数。
- 把 `attention_npu.py` 做成独立封装。
- 在模型里通过配置选择 torch attention 或 npu attention。
- 对比 PyTorch attention 和 NPU attention 的输出 shape。
- 在小 shape 上做粗略数值对比。

## 需要重点记录的内容

在 `docs/notes/ascend_attention_shapes.md` 里记录：

```text
torch_npu version:
CANN version:
device model:
prompt attention API:
incremental attention API:
supported dtype:
supported layout:
q shape:
k shape:
v shape:
output shape:
block table shape:
block size constraints:
known errors:
```

## 验收标准

最低验收：

- `attention_npu.py` 能成功调用一个 Ascend attention 算子。
- 输出 shape 和 PyTorch attention 对齐。
- 模型可以通过配置选择 NPU attention backend。

进阶验收：

- prefill 和 decode 都走 NPU attention。
- paged KV cache 的 block table 能传入 NPU attention。

## 常见坑

- layout 名称理解错，例如 BSH、BNSD、BSND。
- GQA 的 head 数传错。
- K/V cache layout 和 API 要求不一致。
- `block_table` dtype 不对。
- mask dtype 不对。
- context length 没有传真实长度。
- 小 shape 能跑，大 shape 因对齐约束报错。

## 阶段完成后的下一步

进入 Stage 07，把 attention 后端接回完整 batch engine。

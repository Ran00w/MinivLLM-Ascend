# Stage 02：设备后端抽象与 NPU eager

## 学习目标

把 Stage 01 的 tiny 推理从 CPU/PyTorch 扩展到 Ascend NPU eager 模式。

核心不是写高性能代码，而是学会：

- `torch_npu` 如何引入。
- tensor 如何放到 NPU。
- 如何同步 NPU。
- 如何避免业务代码到处写设备专用 API。

## 本阶段不做什么

- 不接 Ascend fused attention。
- 不做 graph。
- 不做多卡。
- 不做真实模型。

## 建议新增文件

```text
src/minivllm_ascend/backend/
  __init__.py
  base.py
  torch_cpu.py
  torch_npu.py
scripts/
  run_tiny_cpu.py
  run_tiny_npu.py
tests/
  test_backend.py
```

## 推荐 backend 接口

```text
Backend
  name
  device
  is_available()
  tensor(data, dtype)
  to_device(tensor)
  synchronize()
  empty_cache()
  memory_info()
```

## 任务清单

- 安装并验证 `torch_npu`。
- 在 `torch_npu.py` 中集中导入 `torch_npu`。
- 实现 CPU backend。
- 实现 NPU backend。
- 改造 `SimpleEngine`，不要直接写 `torch.device("cpu")`。
- 所有 tensor 创建都走 backend 或使用已有 tensor 的 device。
- 跑通 `scripts/run_tiny_cpu.py`。
- 跑通 `scripts/run_tiny_npu.py`。

## 关键检查点

检查代码里是否还有这些硬编码：

```text
.cuda()
torch.cuda
device="cuda"
device="npu"
```

业务代码里不应该直接出现它们。只有 backend 层可以出现。

## 验收标准

同一个 tiny model 可以通过参数选择：

```bash
python scripts/run_tiny.py --device cpu
python scripts/run_tiny.py --device npu
```

并且两条路径都能完成生成循环。

## 常见坑

- 忘记 `import torch_npu`，导致 PyTorch 不认识 NPU backend。
- tensor 一部分在 CPU，一部分在 NPU。
- 用 Python list 频繁和 NPU tensor 互转。
- 在循环里做太多 `.cpu()`，导致同步变慢。

## 阶段完成后的下一步

进入 Stage 03，引入真实模型结构和权重加载。

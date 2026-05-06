# Stage 02：设备后端抽象与 NPU eager

## 这一阶段到底要做成什么

Stage 01 已经跑通了一个 tiny Transformer 的最小生成闭环：

```text
prompt token ids
  -> model
  -> logits
  -> sampler
  -> append token
  -> repeat
```

Stage 02 要做的是：让这条闭环既能跑在 CPU 上，也能跑在 Ascend NPU 的 eager 模式上。

但重点不是把所有地方的 `cpu` 或 `cuda` 字符串简单替换成 `npu`。真正要学的是设备后端抽象：

```text
业务代码:
  SimpleEngine / TinyTransformer / Attention / Sampler

不应该到处关心:
  是 CPU 还是 NPU
  怎么 import torch_npu
  怎么 synchronize
  怎么 empty cache

这些硬件差异应该集中到:
  backend/
```

这一阶段结束后，你应该能通过一个参数选择运行设备：

```bash
PYTHONPATH=src python scripts/run_tiny.py --device cpu
PYTHONPATH=src python scripts/run_tiny.py --device npu
```

两条路径都应该能完成 tiny model 的生成循环。输出 token 不要求一样，性能也不重要。重要的是：代码结构开始像一个可以继续扩展到真实推理引擎的项目。

## 为什么 Stage 02 要先做 backend 抽象

AI infra 项目很容易一开始写成这样：

```python
device = "cuda:0"
input_ids = torch.tensor(..., device=device)
self.linear = nn.Linear(..., device=device)
mask = torch.ones(..., device=device)
torch.cuda.synchronize()
```

如果后面要迁移到 Ascend NPU，你可能会想把它改成：

```python
device = "npu:0"
torch.npu.synchronize()
```

这能短期跑起来，但不是好结构。因为设备专用逻辑会散落到 engine、model、attention、sampler、脚本和测试里。后面 Stage 06 接 Ascend attention、Stage 08 接 graph、Stage 09 接多卡时，会越来越难改。

Stage 02 的核心原则是：

```text
只有 backend 层可以知道具体设备细节。
业务层只拿 backend.device 或调用 backend 方法。
```

也就是说：

- `torch_npu` 的导入集中在 `backend/torch_npu.py`。
- `torch.device("npu:0")` 集中在 NPU backend。
- `torch.npu.synchronize()` 集中在 NPU backend。
- `torch.cuda` 不应该出现在这个 Ascend 项目的业务代码里。
- `device = "cuda:0"` 更不应该留在 Stage 02 代码里。

## 本阶段学习目标

完成本阶段后，你应该能回答这些问题：

- `torch_npu` 为什么通常需要先 import，PyTorch 才认识 NPU 设备。
- `torch.device("cpu")`、`torch.device("npu:0")` 和 backend 对象是什么关系。
- eager 模式是什么意思。
- 为什么不要在 model/layer 里写死 `device="npu:0"`。
- `model.to(device)` 和 `torch.tensor(..., device=device)` 分别解决什么问题。
- 为什么 tensor 的 device 必须一致。
- 哪些地方会触发 NPU 到 CPU 的同步。
- 为什么 backend 抽象会帮助后面的 graph、attention 算子和多卡。

## 本阶段不做什么

这些内容先不要做：

- 不接 Ascend fused attention。
- 不调用 `torch_npu.npu_prompt_flash_attention`。
- 不调用 `torch_npu.npu_incre_flash_attention`。
- 不做 KV cache。
- 不做 paged attention。
- 不做 graph capture。
- 不做 `torch.compile` 优化。
- 不做多卡 HCCL。
- 不加载 Qwen3 真实权重。
- 不追求 NPU 性能。
- 不要求 CPU 和 NPU 生成出完全相同 token。

Stage 02 的目标只有一个：把 Stage 01 的 tiny 推理闭环迁移到 CPU/NPU 可切换的 eager 后端结构上。

## 目录和文件建议

项目总结构文档推荐最终包名是：

```text
src/minivllm_ascend/
```

如果你现在为了学习按阶段建了目录，例如：

```text
src/stage01/
src/stage02/
```

也可以继续沿用。关键是理解模块边界。下面同时给出“最终项目结构”和“按阶段学习结构”的放法。

### 推荐最终项目结构

```text
src/minivllm_ascend/
  backend/
    __init__.py
    base.py
    torch_cpu.py
    torch_npu.py
  engine/
    simple_engine.py
  layers/
    attention_torch.py
    sampler.py
    layernorm.py
  models/
    tiny_transformer.py
  sampling_params.py
scripts/
  run_tiny.py
tests/
  test_backend.py
  test_tiny_generation.py
```

### 如果沿用当前 stage 目录

```text
src/stage02/
  backend/
    __init__.py
    base.py
    torch_cpu.py
    torch_npu.py
  engine/
    engine.py
    sequence.py
  layers/
    attention_torch.py
    model.py
    norm.py
    sampler.py
  sampling_params.py
scripts/
  run.py
tests/
  test_backend.py
```

当前代码里如果已经有这些写法：

```python
device = "cuda:0"
nn.Linear(..., device=device)
torch.ones(..., device=device)
```

Stage 02 的主要任务就是把它们收敛掉。不要把 `cuda:0` 简单替换成 `npu:0`，而是改成从 backend 或已有 tensor 获取 device。

## 推荐实现顺序

按这个顺序做，最不容易乱：

1. 先写 `backend/base.py`，明确 backend 需要提供哪些能力。
2. 写 `backend/torch_cpu.py`，让 CPU 路径先跑通。
3. 写 `backend/torch_npu.py`，集中处理 `torch_npu` 导入和 NPU API。
4. 写 `backend/__init__.py` 或 factory 函数，通过字符串创建 backend。
5. 改 `SimpleEngine`，让 input tensor 创建走 backend.device。
6. 改 tiny model 和 layers，去掉所有全局 `device = "cuda:0"`。
7. 改脚本，支持 `--device cpu` 和 `--device npu`。
8. 写 `test_backend.py`，CPU 必测，NPU 没环境时跳过。
9. 用 `rg` 检查业务代码里是否还残留 CUDA/NPU 硬编码。
10. 分别跑 CPU 和 NPU demo。

## 模块 1：`backend/base.py`

### 这个模块要干什么

`base.py` 定义所有 backend 都应该支持的接口。它不负责具体实现，只负责规定“业务代码能依赖什么能力”。

Stage 02 先把接口设计得小一点。够用就好，不要一开始做复杂运行时系统。

### 推荐接口

```text
Backend
  name: str
  device: torch.device
  is_available() -> bool
  tensor(data, dtype=None) -> torch.Tensor
  to_device(tensor) -> torch.Tensor
  module_to_device(module) -> nn.Module
  synchronize() -> None
  empty_cache() -> None
  memory_info() -> dict[str, int | None]
```

### 推荐代码形态

可以用抽象基类：

```python
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Backend(ABC):
    name: str
    device: torch.device

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    def tensor(
        self,
        data: Any,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=self.device)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def module_to_device(self, module: nn.Module) -> nn.Module:
        return module.to(self.device)

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def memory_info(self) -> dict[str, int | None]:
        return {"allocated": None, "reserved": None}
```

### 为什么要有 `module_to_device`

你可以直接在业务代码里写：

```python
model.to(backend.device)
```

这也可以。但加一个 `module_to_device` 的好处是让迁移模型设备的动作也归到 backend 语义里。后面如果 NPU 需要特殊处理，你不用改所有调用点。

### 新人需要注意

backend 接口不要太大。Stage 02 只需要管理这些基础能力：

- 创建 tensor。
- 移动 tensor。
- 移动 module。
- 同步设备。
- 查询简单内存信息。

不要现在就把 graph、分布式、attention backend、KV cache 都塞进来。那些是后面阶段的内容。

## 模块 2：`backend/torch_cpu.py`

### 这个模块要干什么

CPU backend 是最简单的后端，也是你的兜底验证路径。

它的作用是：

- 不依赖 NPU 环境。
- 保证 Stage 01 的 tiny 推理仍然能跑。
- 让测试在没有 Ascend 机器时也能执行。

### 推荐实现

```python
import torch

from .base import Backend


class TorchCPUBackend(Backend):
    name = "cpu"

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def is_available(self) -> bool:
        return True
```

### CPU backend 为什么也需要

不要觉得项目目标是 Ascend，就只写 NPU backend。CPU backend 很重要：

- 没有 NPU 的机器也能跑单元测试。
- CPU 报错通常更容易读。
- CPU 路径能帮你确认问题是模型逻辑错，还是 NPU 环境错。
- 后面做 NPU 数值对比时，CPU/PyTorch 是参考实现。

## 模块 3：`backend/torch_npu.py`

### 这个模块要干什么

NPU backend 负责集中接入 `torch_npu`。

业务代码不应该到处写：

```python
import torch_npu
torch.npu.synchronize()
torch.device("npu:0")
```

这些都应该放在 `torch_npu.py` 里。

### 推荐实现方式

`torch_npu` 可能在没有 Ascend 环境的机器上导入失败。所以不要在包顶层到处 import 它。推荐只在 NPU backend 里导入，并且把错误信息保留下来。

```python
import torch

from .base import Backend


class TorchNPUBackend(Backend):
    name = "npu"

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self._import_error: Exception | None = None

        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            self._import_error = exc
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"npu:{device_id}")

    def is_available(self) -> bool:
        if self._import_error is not None:
            return False
        return hasattr(torch, "npu") and torch.npu.is_available()

    def synchronize(self) -> None:
        if self.is_available():
            torch.npu.synchronize()

    def empty_cache(self) -> None:
        if self.is_available():
            torch.npu.empty_cache()

    def memory_info(self) -> dict[str, int | None]:
        if not self.is_available():
            return {"allocated": None, "reserved": None}
        return {
            "allocated": torch.npu.memory_allocated(self.device),
            "reserved": torch.npu.memory_reserved(self.device),
        }
```

### 为什么 `import torch_npu` 很关键

在 Ascend PyTorch 环境里，`torch_npu` 通常会向 PyTorch 注册 NPU backend。没有导入时，直接写：

```python
torch.device("npu:0")
```

不一定足够。你可能会遇到 PyTorch 不认识 NPU 设备，或者 `torch.npu` API 不完整的问题。

所以 Stage 02 的规则是：

```text
只有 backend/torch_npu.py 负责 import torch_npu。
其他文件不要 import torch_npu。
```

上面示例里，如果 `torch_npu` 导入失败，临时把 `self.device` 设成 CPU 只是为了让对象能被构造、让 `is_available()` 返回 False。业务代码在使用 NPU backend 前必须先检查 `is_available()`，不要在不可用时继续创建 tensor。

### 如果没有 NPU 环境怎么办

没有 NPU 环境时，CPU 路径必须仍然能跑。

NPU 测试可以跳过，但不能让整个项目 import 失败。也就是说：

- `import stage02` 或 `import minivllm_ascend` 不应该因为没有 `torch_npu` 失败。
- 只有用户选择 `--device npu` 时，才需要检查 `torch_npu` 和 NPU 是否可用。
- 如果不可用，错误信息要清楚告诉用户是 NPU backend 不可用。

## 模块 4：`backend/__init__.py`

### 这个模块要干什么

这里可以放一个简单 factory，根据字符串返回 backend。

推荐支持：

```text
cpu
npu
npu:0
npu:1
```

### 推荐实现

```python
from .base import Backend
from .torch_cpu import TorchCPUBackend
from .torch_npu import TorchNPUBackend


def get_backend(name: str) -> Backend:
    normalized = name.lower()

    if normalized == "cpu":
        return TorchCPUBackend()

    if normalized == "npu":
        return TorchNPUBackend(device_id=0)

    if normalized.startswith("npu:"):
        device_id = int(normalized.split(":", 1)[1])
        return TorchNPUBackend(device_id=device_id)

    raise ValueError(f"unsupported backend: {name}")
```

### 新人需要注意

这个 factory 不应该偷偷 fallback。

例如用户写：

```bash
python scripts/run_tiny.py --device npu
```

但机器没有 NPU，你不应该静默改成 CPU。应该明确报错：

```text
NPU backend is not available. Check torch_npu, CANN, and device visibility.
```

静默 fallback 会让你误以为自己跑在 NPU 上，实际上一直跑 CPU。

## 模块 5：改造 `SimpleEngine`

### 这个模块要改什么

Stage 01 的 engine 里可能有类似代码：

```python
input_ids = torch.tensor(
    [sequence.all_token_ids],
    dtype=torch.long,
    device=next(self.model.parameters()).device,
)
```

或者你当前代码里有：

```python
device = "cuda:0"
input_ids = torch.tensor(..., device=device)
```

Stage 02 要把它改成由 backend 控制：

```python
input_ids = self.backend.tensor(
    [sequence.all_token_ids],
    dtype=torch.long,
)
```

### 推荐接口

```text
SimpleEngine
  __init__(model, sampler, backend, verbose=False)
  generate(prompt_token_ids, sampling_params) -> Sequence
```

### 推荐逻辑

```python
class SimpleEngine:
    def __init__(
        self,
        model: nn.Module,
        sampler: GreedySampler,
        backend: Backend,
        verbose: bool = False,
    ) -> None:
        self.backend = backend
        self.model = backend.module_to_device(model)
        self.sampler = sampler
        self.verbose = verbose

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        if not self.backend.is_available():
            raise RuntimeError(f"backend {self.backend.name} is not available")

        self.model.eval()
        sequence = Sequence(...)

        step = 0
        while not sequence.is_finished:
            input_ids = self.backend.tensor(
                [sequence.all_token_ids],
                dtype=torch.long,
            )

            logits = self.model(input_ids)
            last_token_logits = logits[:, -1, :]
            next_token_ids = self.sampler.sample(last_token_logits)
            next_token_id = int(next_token_ids[0].item())

            phase = "prefill" if step == 0 else "decode"
            if self.verbose:
                print(
                    f"step={step} phase={phase} "
                    f"device={input_ids.device} "
                    f"input_len={input_ids.shape[1]} "
                    f"next_token={next_token_id}"
                )

            sequence.append_token(next_token_id)
            step += 1

        self.backend.synchronize()
        return sequence
```

### 关于 `.item()`

这一行：

```python
next_token_id = int(next_token_ids[0].item())
```

在 NPU 上会触发一次设备同步，因为 Python 需要拿到普通 int 才能 append 到 `Sequence`。

Stage 02 可以接受这个同步，因为现在是学习闭环，不追性能。你要知道它会同步，但不要在这一阶段为了避免它引入复杂结构。

后面真正做 batch engine 时，会更认真地管理设备输出和 CPU 状态。

### 新人需要注意

`@torch.no_grad` 要写成：

```python
@torch.no_grad()
```

不要少了括号。少括号时它不是你以为的执行方式，容易产生奇怪问题。

## 模块 6：改造 tiny model 和 layers

### 这个模块要改什么

Stage 02 里，模型和 layer 不应该再有全局设备硬编码。

不要写：

```python
device = "cuda:0"
self.linear = nn.Linear(hidden_size, hidden_size, device=device)
mask = torch.ones(seq_len, seq_len, device=device)
```

推荐写：

```python
self.linear = nn.Linear(hidden_size, hidden_size)
```

模型创建后统一：

```python
model = backend.module_to_device(model)
```

forward 里需要新建 tensor 时，用已有输入 tensor 的 device：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
position_ids = torch.arange(seq_len, device=input_ids.device)
```

### 为什么不要在 `nn.Linear` 里写 device

如果每一层都写：

```python
nn.Linear(..., device=device)
```

那每个 layer 都要知道设备。后面你改设备、做多卡、做权重加载时都会麻烦。

更推荐：

```text
1. 先在 CPU 上构建模型结构。
2. 调用 model.to(backend.device) 统一移动参数。
3. forward 中新建临时 tensor 时跟随输入 tensor 的 device。
```

这也是 PyTorch 项目里更常见的写法。

### attention mask 要怎么改

Stage 01 里 attention mask 可能这样写：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
```

Stage 02 改成：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
mask = mask.view(1, 1, seq_len, seq_len)
```

这样不管 `x` 在 CPU 还是 NPU，mask 都会创建在同一个设备上。

### position ids 要怎么改

如果 tiny model 有 position embedding，写成：

```python
position_ids = torch.arange(seq_len, device=input_ids.device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
```

不要写：

```python
position_ids = torch.arange(seq_len).to("npu")
```

因为这会把设备逻辑漏到 model 里。

## 模块 7：改造 sampler

### 这个模块要改什么

Stage 02 的 sampler 仍然可以是 greedy：

```python
next_token_ids = torch.argmax(logits, dim=-1)
```

不需要先 softmax。因为：

```text
argmax(softmax(logits)) == argmax(logits)
```

少做 softmax 更简单，也少一个算子。

### 暂时不要用 `torch.compile`

Stage 02 是 eager 后端阶段。建议先不要在 sampler 上加：

```python
@torch.compile
```

原因：

- `torch.compile` 会引入额外编译变量。
- NPU 对 `torch.compile` 的支持和限制会随环境变化。
- 现在你要验证的是设备抽象，不是编译优化。

后面 Stage 08 做 graph 和性能实验时，再系统研究 compile/graph。

## 模块 8：改造脚本

### 脚本要做什么

Stage 02 推荐把 Stage 01 的脚本改成可以选择设备：

```bash
PYTHONPATH=src python scripts/run_tiny.py --device cpu
PYTHONPATH=src python scripts/run_tiny.py --device npu
```

如果你当前只有 `scripts/run.py`，也可以先加参数：

```bash
PYTHONPATH=src python scripts/run.py --device cpu
PYTHONPATH=src python scripts/run.py --device npu
```

### 推荐参数

```text
--device cpu 或 npu
--max-tokens 生成 token 数
--prompt-ids 手写 prompt token ids
--seed 随机种子
--verbose 打印每步日志
```

### 推荐脚本结构

```python
import argparse
import torch

from stage02.backend import get_backend
from stage02.engine.engine import Engine
from stage02.layers.model import transformers
from stage02.layers.sampler import Sampler
from stage02.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    backend = get_backend(args.device)
    if not backend.is_available():
        raise RuntimeError(f"backend {args.device} is not available")

    model = transformers()
    sampler = Sampler()
    params = SamplingParams(max_tokens=args.max_tokens)
    engine = Engine(
        model=model,
        sampler=sampler,
        backend=backend,
        verbose=args.verbose,
    )

    prompt = [1, 4, 9, 10]
    sequence = engine.generate(prompt, params)

    print(f"backend={backend.name} device={backend.device}")
    print(f"prompt_token_ids={sequence.prompt_token_ids}")
    print(f"generated_token_ids={sequence.generated_token_ids}")
    print(f"final_token_ids={sequence.prompt_token_ids + sequence.generated_token_ids}")


if __name__ == "__main__":
    main()
```

如果你已经按 `minivllm_ascend` 包名组织，就把 import 改成对应包名即可。

## 模块 9：`tests/test_backend.py`

### 这个测试要验证什么

backend 测试不需要验证模型输出质量。它要验证设备抽象是否真的工作。

建议至少测：

```text
1. CPU backend 总是可用。
2. CPU backend 创建的 tensor 在 CPU。
3. get_backend("cpu") 返回 CPU backend。
4. get_backend("bad") 会报错。
5. NPU backend 在无环境时不会导致 import 整个项目失败。
6. 如果 NPU 可用，NPU backend 创建的 tensor 在 npu。
```

### CPU backend 测试思路

```python
def test_cpu_backend_tensor_device():
    backend = TorchCPUBackend()
    tensor = backend.tensor([1, 2, 3], dtype=torch.long)

    assert backend.is_available()
    assert tensor.device.type == "cpu"
```

### factory 测试思路

```python
def test_get_backend_cpu():
    backend = get_backend("cpu")
    assert backend.name == "cpu"
    assert backend.device.type == "cpu"


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_backend("cuda")
```

在 Ascend 项目里，`cuda` 不应该是合法 backend。即使你的开发机器有 NVIDIA GPU，也不要让这个项目悄悄走 CUDA 路径。

### NPU 测试要能跳过

```python
def test_npu_backend_tensor_device_if_available():
    backend = TorchNPUBackend()
    if not backend.is_available():
        pytest.skip("NPU backend is not available in this environment")

    tensor = backend.tensor([1, 2, 3], dtype=torch.long)
    assert tensor.device.type == "npu"
```

这样在没有 NPU 的机器上，测试不会失败；在有 NPU 的机器上，又能验证真正设备。

## 模块 10：tiny generation 的设备测试

### 这个测试要验证什么

除了单独测 backend，还要测 tiny generation 能通过 backend 跑。

CPU 必测：

```text
使用 CPU backend
构造 tiny model
跑 generate
确认 generated_token_ids 长度不超过 max_tokens
确认所有模型参数在 CPU
```

NPU 有环境再测：

```text
使用 NPU backend
构造 tiny model
跑 generate
确认 generated_token_ids 长度不超过 max_tokens
确认所有模型参数在 NPU
```

### 参数 device 检查

可以写一个辅助函数：

```python
def assert_module_on_device(module: nn.Module, device_type: str) -> None:
    for parameter in module.parameters():
        assert parameter.device.type == device_type
```

这比只看 `input_ids.device` 更可靠，因为模型参数也可能忘了迁移。

## 环境检查

### 先检查 PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

### 再检查 torch_npu

不要在文档里写死 `torch_npu` 版本。它必须和你的 PyTorch、CANN、驱动环境匹配。你要记录当前环境实际是什么。

可以运行：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)

try:
    import torch_npu
    print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
    print("has torch.npu:", hasattr(torch, "npu"))
    print("npu available:", torch.npu.is_available())
    if torch.npu.is_available():
        print("current device:", torch.npu.current_device())
        print("device count:", torch.npu.device_count())
except Exception as exc:
    print("torch_npu import failed:", repr(exc))
PY
```

如果这一步失败，不要先改模型代码。先确认环境：

```text
PyTorch 版本
torch_npu 版本
CANN 版本
Python 版本
NPU 是否可见
环境变量是否正确
```

### 推荐记录到学习笔记

```text
OS:
Python:
PyTorch:
torch_npu:
CANN:
NPU:
torch.npu.is_available:
```

Stage 02 很多问题不是代码问题，而是环境版本不匹配。把环境记清楚，后面排错会省很多时间。

## 什么是 eager 模式

eager 模式就是普通 PyTorch 执行方式：代码运行到一个 op，就立即调度这个 op。

例如：

```python
y = linear(x)
z = torch.relu(y)
```

在 eager 模式里，`linear` 和 `relu` 会按 Python 执行顺序立即运行。

它和后面的 graph/compile 不同：

```text
eager:
  简单、直观、容易调试。

graph/compile:
  可能更快，但要求 shape 更稳定，也会引入编译、capture、replay 等新问题。
```

Stage 02 必须先用 eager。因为你现在要确认的是：

- tensor 是否在正确设备。
- model 参数是否在正确设备。
- 每一步生成逻辑是否还能跑。
- NPU 环境是否可用。

如果一开始就加 graph 或 compile，报错会混在一起，很难判断是设备问题、模型问题还是编译问题。

## 设备迁移规则

Stage 02 建议遵守这些规则。

### 规则 1：模型统一迁移

推荐：

```python
model = TinyTransformer(config)
model = backend.module_to_device(model)
```

不推荐：

```python
self.linear = nn.Linear(..., device="npu:0")
```

### 规则 2：输入 tensor 由 backend 创建

推荐：

```python
input_ids = backend.tensor([token_ids], dtype=torch.long)
```

不推荐：

```python
input_ids = torch.tensor([token_ids]).to("npu")
```

### 规则 3：forward 里的临时 tensor 跟随输入

推荐：

```python
mask = torch.ones(seq_len, seq_len, device=x.device)
position_ids = torch.arange(seq_len, device=input_ids.device)
```

不推荐：

```python
mask = torch.ones(seq_len, seq_len, device="npu")
```

### 规则 4：不要混用 CPU 和 NPU tensor

这类代码很容易报错：

```python
x = torch.randn(1, 4, 64, device="npu")
mask = torch.ones(4, 4)  # CPU
x = x.masked_fill(mask == 0, float("-inf"))
```

mask 默认在 CPU，`x` 在 NPU，设备不一致。

正确写法：

```python
mask = torch.ones(4, 4, device=x.device)
```

### 规则 5：不要在循环里频繁 `.cpu()`

这会触发同步和拷贝：

```python
logits_cpu = logits.cpu()
```

Stage 02 里只有取 `next_token_id.item()` 这一处同步可以接受。不要为了打印，把整个 logits 搬回 CPU。

## 关键检查点

Stage 02 写完后，用 `rg` 检查：

```bash
rg -n "cuda|\\.cuda\\(|torch\\.cuda|device *= *[\"']cuda|device *= *[\"']npu|torch_npu|torch\\.npu" src scripts tests
```

你应该看到：

```text
允许出现:
  backend/torch_npu.py 里的 torch_npu
  backend/torch_npu.py 里的 torch.npu
  脚本参数说明里的 npu 字符串
  测试 NPU backend 时的 npu 字符串

不应该出现:
  engine 里写死 cuda/npu
  model 里写死 cuda/npu
  attention 里写死 cuda/npu
  sampler 里写死 cuda/npu
  layernorm 里写死 cuda/npu
```

如果当前代码里还有：

```python
device = "cuda:0"
```

那 Stage 02 还没有完成。

## 任务清单

按这个清单完成即可进入验收：

- 新增 `backend/base.py`。
- 新增 `backend/torch_cpu.py`。
- 新增 `backend/torch_npu.py`。
- 新增 `backend/__init__.py` 或 `get_backend`。
- 确认 CPU backend 可以创建 tensor。
- 确认 NPU backend 在无 NPU 环境时不会导致全项目 import 失败。
- 确认 NPU backend 在有 NPU 环境时 `is_available()` 为 True。
- 改造 `SimpleEngine`，接收 backend。
- 改造 `SimpleEngine`，用 backend 创建 `input_ids`。
- 改造 `SimpleEngine`，在初始化时把 model 移到 backend device。
- 移除 engine 里的 `device = "cuda:0"`。
- 移除 model/layer 里的 `device = "cuda:0"`。
- 移除 `nn.Linear(..., device=device)` 这类硬编码。
- attention mask 使用 `x.device`。
- position ids 使用 `input_ids.device`。
- sampler 暂时不要使用 `torch.compile`。
- 脚本支持 `--device cpu`。
- 脚本支持 `--device npu`。
- 写 `test_backend.py`。
- CPU 路径测试通过。
- 有 NPU 环境时 NPU 路径 demo 能跑。
- 用 `rg` 检查没有业务层设备硬编码。
- 写 Stage 02 学习记录，记录环境和报错。

## 推荐调试顺序

如果 NPU 跑不通，按这个顺序查：

1. 先运行环境检查脚本，确认 `import torch_npu` 成功。
2. 确认 `torch.npu.is_available()` 是 True。
3. 单独创建 NPU tensor：`torch.tensor([1], device="npu:0")`。
4. 单独创建 CPU backend tensor，确认 CPU 路径没坏。
5. 单独创建 NPU backend tensor，确认 backend 没坏。
6. 构造 tiny model 后，检查第一层参数的 `parameter.device`。
7. 在 engine 里打印 `input_ids.device`。
8. 在 attention 里临时打印 `x.device` 和 `mask.device`。
9. 如果报 device mismatch，优先找 forward 里新建 tensor 的位置。
10. 如果报 unsupported op，先缩小模型 shape，再确认是不是某个 PyTorch op 在当前 NPU 环境不支持。

不要一边改 backend，一边改模型结构，一边改采样策略。一次只排一个变量。

## 常见坑

- 只把 input_ids 放到 NPU，模型参数还在 CPU。
- 模型参数在 NPU，但 attention mask 默认创建在 CPU。
- `torch_npu` 没导入，导致 NPU backend 不可用。
- 在没有 NPU 的机器上 import 包就失败，因为顶层直接 import 了 `torch_npu`。
- 把 `cuda:0` 改成 `npu:0`，但硬编码问题还在。
- 在 sampler 上用了 `torch.compile`，导致 eager 阶段混入编译问题。
- 为了打印日志，把 logits 整个 `.cpu()`。
- NPU 不可用时脚本静默 fallback 到 CPU，导致误判。
- 测试里强制要求 NPU 存在，导致没有 Ascend 环境的机器无法跑基础测试。
- CPU 和 NPU 随机初始化不同，错误地断言二者生成 token 必须完全一致。

## 验收标准

### 1. CPU demo 能跑

```bash
PYTHONPATH=src python scripts/run_tiny.py --device cpu --max-tokens 8 --verbose
```

或者如果你当前脚本是 `scripts/run.py`：

```bash
PYTHONPATH=src python scripts/run.py --device cpu --max-tokens 8 --verbose
```

应该看到：

- backend 是 CPU。
- input tensor 在 CPU。
- 每一步能打印 `prefill` 或 `decode`。
- 生成 token 数不超过 `max_tokens`。

### 2. NPU demo 能跑

在有 Ascend NPU 环境时运行：

```bash
PYTHONPATH=src python scripts/run_tiny.py --device npu --max-tokens 8 --verbose
```

或者：

```bash
PYTHONPATH=src python scripts/run.py --device npu --max-tokens 8 --verbose
```

应该看到：

- backend 是 NPU。
- input tensor 在 `npu:0`。
- 模型参数在 NPU。
- 能完成生成循环。

如果没有 NPU 环境，脚本应该明确报 NPU backend 不可用，而不是静默跑 CPU。

### 3. backend 测试能跑

```bash
PYTHONPATH=src pytest tests/test_backend.py
```

CPU backend 测试必须通过。

NPU backend 测试规则：

- 有 NPU 环境：应该通过。
- 没有 NPU 环境：应该 skip，而不是 fail。

### 4. 硬编码检查通过

```bash
rg -n "cuda|\\.cuda\\(|torch\\.cuda|device *= *[\"']cuda|device *= *[\"']npu|torch_npu|torch\\.npu" src scripts tests
```

结果里不应该有业务代码硬编码设备。

## 学习记录建议

完成后，在 Stage 02 学习记录里写这些内容：

```text
本阶段目标:
  把 Stage 01 tiny 推理闭环改造成 CPU/NPU backend 可切换。

当前调用链:
  scripts/run_tiny.py --device npu
    -> get_backend("npu")
    -> TorchNPUBackend
    -> SimpleEngine.generate
    -> backend.tensor(...)
    -> TinyTransformer.forward
    -> Sampler.sample
    -> Sequence.append_token

环境:
  OS:
  Python:
  PyTorch:
  torch_npu:
  CANN:
  NPU:
  torch.npu.is_available:

关键 device:
  backend.device:
  input_ids.device:
  first model parameter device:
  attention mask device:
  logits.device:
  next_token_ids.device:

当前限制:
  仍然是 tiny 随机模型。
  仍然没有 tokenizer。
  仍然没有真实模型权重。
  仍然没有 KV cache。
  NPU 路径只是 eager 普通 PyTorch op。
  没有使用 Ascend fused attention。

遇到的问题:
  记录 import torch_npu、device mismatch、unsupported op、环境变量等问题。

下一阶段为什么要做:
  Stage 03 要引入真实 Qwen3 模型结构和权重加载。只有 backend 抽象稳定后，真实模型才不会到处写设备专用代码。
```

## 阶段完成后的下一步

进入 Stage 03：Qwen3 模型结构与权重加载。

Stage 02 解决的是“设备能切换、NPU eager 能跑”的问题，但模型还是 tiny 随机模型。Stage 03 要解决的是：

- 真实模型结构怎么写。
- Qwen3 的 RMSNorm、RoPE、MLP、QKV projection 怎么实现。
- safetensors 权重怎么加载。
- 权重加载后如何继续复用 Stage 02 的 backend，把模型移动到 CPU 或 NPU。

进入 Stage 03 前，先确认：

- CPU 路径稳定。
- NPU 环境检查清楚。
- NPU eager tiny demo 能跑，或者没有 NPU 时能明确 skip。
- 业务代码里没有 `cuda:0`。
- `torch_npu` 只在 backend 层出现。
- 你能解释 backend、device、eager 三者的关系。

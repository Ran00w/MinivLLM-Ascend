# Stage 01：最小 PyTorch 推理闭环

## 这一阶段到底要做成什么

这一阶段先不碰真实大模型，也不碰 NPU。你要先在 CPU 上写出一个最小的 decoder-only Transformer，然后让它完成一次“输入 token ids，循环生成新 token ids”的完整闭环。

最小闭环长这样：

```text
prompt token ids
  -> Sequence 保存请求状态
  -> TinyTransformer forward
  -> 得到 logits
  -> Sampler 选择 next token
  -> Sequence.append_token
  -> 判断是否停止
  -> 继续下一轮
```

这一阶段生成出来的 token 不需要像自然语言。模型可以随机初始化，输出可以很奇怪。重要的是把推理引擎最核心的控制流写清楚：

- prompt 如何进入模型。
- 模型输出的 logits 是什么 shape。
- 为什么只取最后一个位置的 logits 来生成下一个 token。
- sampler 如何从 logits 里选出 token id。
- 生成出来的 token 如何追加回 sequence。
- 循环什么时候停。
- prefill 和 decode 在概念上有什么区别。

你现在是从 AI infra 入门，所以这一阶段不要追求“像 ChatGPT 一样输出文本”。这一阶段的合格标准是：你能看着日志解释每一步输入了几个 token，输出了哪个新 token，为什么循环停下。

## 本阶段学习目标

完成本阶段后，你应该能回答这些问题：

- `input_ids` 是什么，它和自然语言文本有什么关系。
- `hidden_states` 是什么，它为什么有 hidden size。
- attention 为什么要做 causal mask。
- `logits[:, -1, :]` 为什么代表“下一个 token 的预测分布”。
- greedy sampling 为什么等价于 `argmax`。
- `max_tokens` 限制的是生成 token 数，不是 prompt 总长度。
- Stage 01 为什么还没有真正的高效 decode。
- Stage 04 的 KV cache 到底是为了解决 Stage 01 的哪个低效点。

## 本阶段不做什么

这些内容都先不要做，否则你会同时被太多概念卡住：

- 不接 NPU。
- 不导入 `torch_npu`。
- 不加载 Qwen3、Llama 或其他真实大模型权重。
- 不接 tokenizer，直接手写 token id 列表。
- 不做 KV cache。
- 不做 paged attention。
- 不做 batch scheduler。
- 不做多 prompt batch。
- 不做 OpenAI API server。
- 不追求输出文本可读。
- 不追求性能。

## 推荐新增或修改的文件

当前项目里 `src/` 还是空壳，Stage 01 建议先补这些文件：

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

文件命名提醒：

- 文档统一使用 `tiny_transformer.py`，和 `docs/01_architecture_map.md` 保持一致。
- 如果你本地已经建了空的 `tiny_transformers.py`，建议改成单数 `tiny_transformer.py`，避免后面导入路径越来越乱。
- 文档统一使用 `layernorm.py`。如果你已经有 `norm.py`，可以后面再合并，但 Stage 01 先保持名字清楚。

## 推荐实现顺序

不要一上来写完整模型。按下面顺序写，每一步都能单独验证：

1. 写 `SamplingParams`：先定义生成参数。
2. 写 `Sequence`：先把 prompt、生成 token、停止状态管理好。
3. 写 `Sampler`：先实现 greedy sampling。
4. 写 `CausalSelfAttention`：只验证 attention 输入输出 shape。
5. 写 `TinyTransformer`：把 embedding、attention、MLP、lm head 串起来。
6. 写 `SimpleEngine`：把 model、sampler、sequence 串成生成循环。
7. 写 `scripts/run_tiny.py`：打印每一步生成过程。
8. 写测试：先测状态变化，再测生成长度。

这样做的原因是：推理引擎最容易错的不是某一行 PyTorch，而是“状态什么时候更新、长度怎么算、取哪个 logits”。先把外围状态写清楚，再写模型，会更容易定位问题。

## 模块 1：`sampling_params.py`

### 这个模块要干什么

`SamplingParams` 保存“这次请求要怎么生成”的参数。Stage 01 先只需要最少字段，不要一开始把 vLLM 的所有采样参数都搬进来。

建议先支持：

```text
max_tokens: 最多生成多少个新 token
eos_token_id: 遇到哪个 token 就停止，可以先允许为 None
temperature: 先保留字段，但 Stage 01 可以只支持 greedy
```

### 推荐设计

```text
SamplingParams
  max_tokens: int
  eos_token_id: int | None
  temperature: float
```

建议用 `dataclasses.dataclass`，因为它简单、可读：

```python
from dataclasses import dataclass


@dataclass
class SamplingParams:
    max_tokens: int = 16
    eos_token_id: int | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
```

### 新人需要注意

`max_tokens` 指的是“最多生成的新 token 数”，不是 `prompt + output` 的总长度。

例如：

```text
prompt_token_ids = [10, 20, 30]
max_tokens = 5
最终最多允许:
[10, 20, 30, x1, x2, x3, x4, x5]
总长度最多是 8
```

这个区别后面写 scheduler、KV cache、block manager 都会用到。

## 模块 2：`engine/sequence.py`

### 这个模块要干什么

`Sequence` 表示一个正在生成的请求。哪怕 Stage 01 只有单 prompt，也建议现在就引入 `Sequence`，因为后面 Stage 07 的 scheduler 会围绕它工作。

它需要保存：

- 这个请求的 id。
- prompt token ids。
- 已经生成出来的 token ids。
- 当前是否结束。
- 本请求的 `SamplingParams`。

### 推荐字段

```text
Sequence
  seq_id: int
  prompt_token_ids: list[int]
  sampling_params: SamplingParams
  output_token_ids: list[int]
  finished: bool
```

建议提供这些方法或属性：

```text
all_token_ids:
  返回 prompt_token_ids + output_token_ids

num_prompt_tokens:
  prompt 长度

num_generated_tokens:
  已生成 token 数

append_token(token_id):
  追加一个新 token，并判断是否 finished

is_finished:
  返回 finished
```

### `append_token` 应该怎么做

逻辑顺序建议是：

```text
1. 如果 sequence 已经 finished，直接报错或忽略。
2. 把 token_id 转成 int。
3. 追加到 output_token_ids。
4. 如果 token_id == eos_token_id，finished = True。
5. 如果 len(output_token_ids) >= max_tokens，finished = True。
```

伪代码：

```python
def append_token(self, token_id: int) -> None:
    if self.finished:
        raise RuntimeError("cannot append token to a finished sequence")

    token_id = int(token_id)
    self.output_token_ids.append(token_id)

    if self.sampling_params.eos_token_id is not None:
        if token_id == self.sampling_params.eos_token_id:
            self.finished = True

    if len(self.output_token_ids) >= self.sampling_params.max_tokens:
        self.finished = True
```

### 新人需要注意

不要把生成 token 直接 append 到 `prompt_token_ids` 里。推荐把 prompt 和 output 分开保存：

```text
prompt_token_ids: 原始输入，不要改
output_token_ids: 模型生成的新 token
all_token_ids: 每次需要喂模型时临时拼起来
```

这样你后面调试时能清楚地区分：

- 用户输入有多长。
- 模型生成了多长。
- `max_tokens` 有没有按“生成长度”判断。

## 模块 3：`layers/sampler.py`

### 这个模块要干什么

`Sampler` 负责把模型输出的 logits 变成下一个 token id。

Stage 01 先实现 greedy sampling：

```text
next_token = logits.argmax(dim=-1)
```

### 输入输出

sampler 不应该接收整个 `[batch, seq_len, vocab_size]` 的 logits。它只需要最后一个位置的 logits：

```text
last_token_logits: [batch, vocab_size]
next_token_ids: [batch]
```

原因是 decoder-only 语言模型的第 `t` 个位置输出，用来预测第 `t + 1` 个 token。生成下一个 token 时，只关心当前序列最后一个位置。

### 推荐实现

```python
class GreedySampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, vocab_size]")
        return torch.argmax(logits, dim=-1)
```

### 新人需要注意

如果模型 forward 返回：

```text
logits: [batch, seq_len, vocab_size]
```

engine 里应该先取：

```python
last_token_logits = logits[:, -1, :]
```

再交给 sampler。

不要对所有 `seq_len` 位置一起 argmax。那会得到每个历史位置各自预测的 token，不是当前要追加的下一个 token。

## 模块 4：`layers/attention_torch.py`

### 这个模块要干什么

`attention_torch.py` 实现最朴素的 PyTorch causal self-attention。它的目标是正确和易懂，不是快。

Stage 01 不做 KV cache，所以每一步生成时都把完整上下文重新输入模型：

```text
prompt + generated tokens -> full attention -> next token
```

这很慢，但非常适合学习。后面 Stage 04 的 KV cache 就是为了避免每步重复计算全部历史 token。

### 输入输出 shape

假设：

```text
batch = B
seq_len = T
hidden_size = C
num_heads = H
head_dim = D = C / H
```

attention 输入输出：

```text
hidden_states: [B, T, C]
q: [B, H, T, D]
k: [B, H, T, D]
v: [B, H, T, D]
attention_scores: [B, H, T, T]
attention_probs: [B, H, T, T]
attention_output: [B, T, C]
```

### 具体怎么做

一个最小 causal self-attention 包含这些步骤：

```text
1. 用一个 Linear 把 hidden_states 投影成 qkv。
2. 把 qkv 按最后一维切成 q、k、v。
3. reshape 成多头格式：[B, H, T, D]。
4. 计算 attention_scores = q @ k^T / sqrt(D)。
5. 构造 causal mask，禁止当前位置看未来 token。
6. 对 masked scores 做 softmax。
7. attention_probs @ v 得到每个 head 的输出。
8. transpose + reshape 回 [B, T, C]。
9. 过一个输出 Linear。
```

### causal mask 是什么

语言模型生成第 `t` 个位置时，只能看 `0..t` 的 token，不能看未来。

如果 `T = 4`，允许看的位置是：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

PyTorch 里可以这样构造：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
mask = mask.view(1, 1, seq_len, seq_len)
attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
```

### 新人需要注意

常见错误：

- 忘记除以 `sqrt(head_dim)`，softmax 可能变得很尖。
- mask 建在 CPU，但 hidden_states 在其他 device，后面 Stage 02 会报 device mismatch。
- `transpose` 后直接 `view`，可能因为 tensor 不 contiguous 报错。建议用 `.contiguous().view(...)` 或 `.reshape(...)`。
- causal mask 方向写反，导致当前 token 只能看未来，生成逻辑完全错。

## 模块 5：`layers/layernorm.py`

### 这个模块要干什么

Stage 01 可以直接用 `torch.nn.LayerNorm`。单独建 `layernorm.py` 的意义是给后面 Stage 03 的 RMSNorm 留位置。

推荐先写一个非常薄的包装：

```python
class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)
```

### 为什么现在不用 RMSNorm

真实 Qwen3 通常会用 RMSNorm，但 Stage 01 的 tiny model 不需要和真实模型结构完全一致。现在先用 `LayerNorm`，因为它是 PyTorch 内置模块，容易理解。

到 Stage 03 再替换或新增 RMSNorm。

## 模块 6：`models/tiny_transformer.py`

### 这个模块要干什么

实现一个很小的 decoder-only Transformer。它不需要加载权重，不需要 tokenizer，不需要输出可读文本。

它的职责是：

- 接收 `input_ids`。
- 把 token id 变成 embedding。
- 加 position embedding。
- 经过若干层 Transformer block。
- 输出每个位置的 logits。

### 推荐配置类

建议先定义一个 `TinyTransformerConfig`：

```text
vocab_size: 小词表大小，例如 128
hidden_size: hidden 维度，例如 64
num_layers: 层数，例如 2
num_heads: attention head 数，例如 4
intermediate_size: MLP 中间维度，例如 256
max_position_embeddings: 最大上下文长度，例如 128
```

用很小的值，CPU 上跑得快，shape 也容易看。

注意：

```text
hidden_size 必须能被 num_heads 整除
head_dim = hidden_size // num_heads
```

### 推荐模型结构

最小 decoder block：

```text
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

整个模型：

```text
input_ids
  -> token_embedding
  -> position_embedding
  -> decoder block 1
  -> decoder block 2
  -> final layer norm
  -> lm_head
  -> logits
```

### 输入输出 shape

```text
input_ids: [batch, seq_len]
token_embeddings: [batch, seq_len, hidden_size]
position_embeddings: [batch, seq_len, hidden_size]
hidden_states: [batch, seq_len, hidden_size]
logits: [batch, seq_len, vocab_size]
```

### position ids 怎么做

Stage 01 可以先用绝对位置 embedding：

```python
batch_size, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, device=input_ids.device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
```

然后：

```python
hidden_states = token_embedding(input_ids) + position_embedding(position_ids)
```

后面 Stage 03 真实 Qwen3 会换成 RoPE。现在先不用 RoPE，降低难度。

### lm_head 是什么

`lm_head` 是最后一个线性层：

```text
[batch, seq_len, hidden_size] -> [batch, seq_len, vocab_size]
```

它输出每个位置对整个词表的打分。比如 `vocab_size = 128`，最后一维就是 128 个 token 的分数。

### 新人需要注意

模型 forward 应该返回所有位置的 logits：

```text
logits: [B, T, V]
```

不要在模型内部只返回最后一个位置。原因是模型本身只负责计算，`SimpleEngine` 才负责决定“生成时取最后 token 的 logits”。保持这个边界后，后面测试和扩展更清晰。

## 模块 7：`engine/simple_engine.py`

### 这个模块要干什么

`SimpleEngine` 是 Stage 01 的最小推理引擎。它把前面的模块串起来：

```text
Sequence -> model -> sampler -> Sequence
```

它不需要复杂 scheduler，也不需要 batch。先只支持一个 prompt。

### 推荐接口

```text
SimpleEngine
  __init__(model, sampler)
  generate(prompt_token_ids, sampling_params) -> Sequence
```

`generate` 返回 `Sequence`，而不是只返回 list。这样你能看到：

- 原始 prompt。
- 生成 token。
- 是否 finished。
- 最终 all_token_ids。

### 生成循环怎么写

推荐把第一步叫做 prefill：

```text
prefill:
  输入完整 prompt
  得到 prompt 最后位置 logits
  采样第一个新 token
```

后面的步骤叫做 decode：

```text
decode:
  输入 prompt + 已生成 token
  得到最后位置 logits
  采样下一个新 token
```

注意：Stage 01 只是“概念上区分 prefill/decode”。因为还没有 KV cache，所以 decode 阶段仍然会重新输入完整上下文。真正高效的 decode 要到 Stage 04 才做。

伪代码：

```python
class SimpleEngine:
    def __init__(self, model: nn.Module, sampler: GreedySampler) -> None:
        self.model = model
        self.sampler = sampler

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        sequence = Sequence(
            seq_id=0,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        self.model.eval()

        step = 0
        while not sequence.is_finished:
            input_ids = torch.tensor(
                [sequence.all_token_ids],
                dtype=torch.long,
                device=next(self.model.parameters()).device,
            )

            logits = self.model(input_ids)
            last_token_logits = logits[:, -1, :]
            next_token_ids = self.sampler.sample(last_token_logits)
            next_token_id = int(next_token_ids[0].item())

            phase = "prefill" if step == 0 else "decode"
            print(
                f"step={step} phase={phase} "
                f"input_len={input_ids.shape[1]} next_token={next_token_id}"
            )

            sequence.append_token(next_token_id)
            step += 1

        return sequence
```

### 为什么循环条件是 `not sequence.is_finished`

停止逻辑应该集中在 `Sequence.append_token` 里，而不是散落在 engine 里。这样后面有更多停止条件时，比如 EOS、stop token、max length，状态仍然好管理。

### 新人需要注意

每一步都重新创建 `input_ids` tensor 是低效的，但 Stage 01 可以接受。现在先保证逻辑正确。后面优化路线是：

```text
Stage 02: input_ids 可以放到 NPU
Stage 04: decode 只输入最后一个 token
Stage 05: KV cache 改成 paged cache
Stage 07: 多个 sequence batch 调度
```

## 模块 8：`scripts/run_tiny.py`

### 这个脚本要干什么

脚本用于手工验证 Stage 01 的完整闭环。它应该越小越好，只做一件事：跑通 tiny model 生成。

建议脚本做这些事：

```text
1. 设置 torch.manual_seed，方便每次输出一致。
2. 构造 TinyTransformerConfig。
3. 构造 TinyTransformer。
4. 构造 GreedySampler。
5. 构造 SimpleEngine。
6. 手写 prompt token ids。
7. 设置 SamplingParams。
8. 调用 generate。
9. 打印 prompt、每一步 next token、最终 token ids。
```

### 推荐配置

```text
vocab_size = 128
hidden_size = 64
num_layers = 2
num_heads = 4
intermediate_size = 256
max_position_embeddings = 128
prompt_token_ids = [1, 5, 10, 20]
max_tokens = 8
```

### 推荐运行方式

如果项目还没有 `pyproject.toml`，直接运行脚本时 Python 可能找不到 `src/minivllm_ascend`。可以先这样跑：

```bash
PYTHONPATH=src python scripts/run_tiny.py
```

如果后面补了包配置并安装为 editable package：

```bash
pip install -e .
python scripts/run_tiny.py
```

### 期望看到什么

输出类似这样就可以：

```text
prompt_token_ids=[1, 5, 10, 20]
step=0 phase=prefill input_len=4 next_token=37
step=1 phase=decode input_len=5 next_token=81
step=2 phase=decode input_len=6 next_token=12
...
final_token_ids=[1, 5, 10, 20, 37, 81, 12, ...]
generated_token_ids=[37, 81, 12, ...]
```

token 数字不需要和这里一样。随机初始化模型只要流程正确即可。

## 模块 9：`tests/test_sequence.py`

### 这个测试要验证什么

`Sequence` 是后面 engine、scheduler、KV cache 都会依赖的基础状态对象，所以要先测它。

建议至少测三个点：

```text
1. append_token 后 output_token_ids 增加。
2. all_token_ids 等于 prompt + output。
3. 生成 token 数达到 max_tokens 后 finished = True。
```

### 测试样例思路

```python
def test_sequence_append_token():
    params = SamplingParams(max_tokens=2)
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3], sampling_params=params)

    seq.append_token(4)

    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.output_token_ids == [4]
    assert seq.all_token_ids == [1, 2, 3, 4]
    assert not seq.is_finished

    seq.append_token(5)

    assert seq.output_token_ids == [4, 5]
    assert seq.is_finished
```

再加一个 EOS 测试：

```python
def test_sequence_stops_on_eos():
    params = SamplingParams(max_tokens=10, eos_token_id=2)
    seq = Sequence(seq_id=0, prompt_token_ids=[1], sampling_params=params)

    seq.append_token(2)

    assert seq.is_finished
```

## 模块 10：`tests/test_tiny_generation.py`

### 这个测试要验证什么

这个测试不验证生成内容好不好，只验证生成循环的边界条件正确。

建议至少测：

```text
1. generate 能返回 Sequence。
2. 生成数量不超过 max_tokens。
3. 最终 all_token_ids 长度不超过 prompt_len + max_tokens。
4. output_token_ids 长度大于 0。
```

### 测试样例思路

```python
def test_tiny_generation_does_not_exceed_max_tokens():
    torch.manual_seed(0)
    config = TinyTransformerConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    model = TinyTransformer(config)
    engine = SimpleEngine(model=model, sampler=GreedySampler())

    prompt = [1, 2, 3]
    params = SamplingParams(max_tokens=4)
    seq = engine.generate(prompt, params)

    assert len(seq.output_token_ids) <= 4
    assert len(seq.all_token_ids) <= len(prompt) + 4
    assert seq.is_finished
```

如果测试日志太吵，可以给 `SimpleEngine` 加一个 `verbose: bool = False` 参数，让脚本打开日志，测试关闭日志。

## 可选测试：attention shape

虽然阶段任务看板只要求 `test_sequence.py` 和 `test_tiny_generation.py`，但你如果刚开始学 AI infra，强烈建议加一个 attention shape 测试，因为后面绝大多数 bug 都是 shape 错。

测试目标：

```text
输入 hidden_states: [2, 5, 16]
输出 attention_output: [2, 5, 16]
```

还可以检查 causal mask 不允许看未来，但 Stage 01 可以先只测 shape。

## 关键 shape 总结

Stage 01 一定要把这些 shape 记熟：

```text
input_ids: [batch, seq_len]
token_embeddings: [batch, seq_len, hidden_size]
position_embeddings: [batch, seq_len, hidden_size]
hidden_states: [batch, seq_len, hidden_size]

q: [batch, num_heads, seq_len, head_dim]
k: [batch, num_heads, seq_len, head_dim]
v: [batch, num_heads, seq_len, head_dim]

attention_scores: [batch, num_heads, seq_len, seq_len]
attention_probs: [batch, num_heads, seq_len, seq_len]
attention_output: [batch, seq_len, hidden_size]

logits: [batch, seq_len, vocab_size]
last_token_logits: [batch, vocab_size]
next_token_ids: [batch]
```

一个具体例子：

```text
batch = 1
seq_len = 4
hidden_size = 64
num_heads = 4
head_dim = 16
vocab_size = 128

input_ids: [1, 4]
hidden_states: [1, 4, 64]
q/k/v: [1, 4, 4, 16]
attention_scores: [1, 4, 4, 4]
logits: [1, 4, 128]
last_token_logits: [1, 128]
next_token_ids: [1]
```

## prefill 和 decode 在本阶段怎么理解

### prefill

prefill 是处理 prompt 的阶段。

Stage 01 里：

```text
input_ids = prompt_token_ids
logits = model(input_ids)
last_token_logits = logits[:, -1, :]
next_token = sampler(last_token_logits)
```

如果 prompt 长度是 4：

```text
input_ids: [1, 4]
logits: [1, 4, vocab_size]
取 logits[:, 3, :] 采样第一个生成 token
```

### decode

decode 是已经有生成 token 之后，继续生成下一个 token 的阶段。

Stage 01 里没有 KV cache，所以 decode 仍然重新输入完整上下文：

```text
input_ids = prompt_token_ids + output_token_ids
logits = model(input_ids)
last_token_logits = logits[:, -1, :]
next_token = sampler(last_token_logits)
```

这不是高效实现，但概念上已经能看出：

```text
prefill: 用 prompt 生成第一个 token
decode: 用 prompt + 已生成内容继续生成
```

Stage 04 会把 decode 改成：

```text
只输入最新 token + 读取历史 KV cache
```

## 最小任务清单

按这个清单完成即可进入验收：

- 实现 `SamplingParams`。
- 实现 `Sequence`。
- 实现 `Sequence.append_token`。
- 实现 greedy sampler。
- 实现 causal self-attention。
- 实现 tiny token embedding。
- 实现 tiny position embedding。
- 实现 tiny MLP。
- 实现 tiny decoder block。
- 实现 tiny `lm_head`。
- 实现 `TinyTransformer.forward(input_ids)`。
- 实现 `SimpleEngine.generate`。
- 写 `scripts/run_tiny.py`。
- 写 `tests/test_sequence.py`。
- 写 `tests/test_tiny_generation.py`。
- 运行脚本并能解释日志。
- 运行测试并通过。
- 写一篇 Stage 01 学习记录。

## 推荐调试顺序

如果你写完后跑不通，不要乱改。按这个顺序查：

1. 先单独创建 `Sequence`，调用 `append_token`，确认状态变化正确。
2. 单独调用 `GreedySampler.sample(torch.randn(1, vocab_size))`，确认返回 `[1]`。
3. 单独调用 `CausalSelfAttention(torch.randn(B, T, C))`，确认输出 `[B, T, C]`。
4. 单独调用 `TinyTransformer(input_ids)`，确认输出 `[B, T, V]`。
5. 在 `SimpleEngine.generate` 里打印 `input_ids.shape` 和 `logits.shape`。
6. 确认取的是 `logits[:, -1, :]`。
7. 确认 `Sequence.append_token` 后长度增加。
8. 确认达到 `max_tokens` 后循环停止。

## 常见坑

- `hidden_size % num_heads != 0`，导致 reshape 出错。
- `position_ids` 建在 CPU，但模型后面迁移设备后会 device mismatch。
- mask shape 不能 broadcast 到 `[B, H, T, T]`。
- sampler 对 `[B, T, V]` 直接 argmax，导致返回 shape 错。
- 忘记 `model.eval()`，虽然 Stage 01 没有 dropout 也建议养成习惯。
- 忘记 `torch.no_grad()`，导致生成时构建无用计算图。
- `max_tokens` 按总长度判断，导致 prompt 长一点时直接不生成。
- EOS token 和普通 token id 冲突，导致第一步就停。可以先把 `eos_token_id=None`。
- 测试里随机模型输出不稳定，却去断言具体 token id。Stage 01 测试只断言长度、shape、状态。

## 验收标准

### 1. 运行 demo

如果还没有包安装配置：

```bash
PYTHONPATH=src python scripts/run_tiny.py
```

如果已经安装 editable package：

```bash
python scripts/run_tiny.py
```

应该看到：

- 输入 prompt token ids。
- 每一步的 `step`。
- 每一步的 `phase`，第一步是 `prefill`，后面是 `decode`。
- 每一步的 `input_len`。
- 每一步生成的 `next_token`。
- 最终 token ids。
- generated token ids。

### 2. 运行测试

```bash
PYTHONPATH=src pytest tests/test_sequence.py tests/test_tiny_generation.py
```

测试应该验证：

- `Sequence.append_token` 正确追加 token。
- `Sequence` 能在 `max_tokens` 达到时结束。
- 如果设置 EOS，遇到 EOS 能结束。
- `SimpleEngine.generate` 不会生成超过 `max_tokens` 的 token。

### 3. 能口头解释这条链路

你应该能解释：

```text
prompt ids -> input_ids tensor -> model logits -> last_token_logits
-> sampler -> next_token -> append 到 sequence -> 继续循环
```

如果这条链路解释不清楚，不建议进入 Stage 02。

## 学习记录建议

完成后，在阶段学习记录里写这些内容：

```text
本阶段目标:
  跑通一个 CPU PyTorch tiny model 的最小生成循环。

当前调用链:
  scripts/run_tiny.py
    -> SimpleEngine.generate
    -> TinyTransformer.forward
    -> CausalSelfAttention.forward
    -> GreedySampler.sample
    -> Sequence.append_token

关键 shape:
  input_ids:
  hidden_states:
  q/k/v:
  attention_scores:
  logits:
  last_token_logits:
  next_token_ids:

当前限制:
  没有 tokenizer。
  没有真实模型权重。
  没有 NPU。
  没有 KV cache。
  decode 每一步仍然重算完整上下文。

下一阶段为什么要做:
  Stage 02 要把这条闭环迁移到 NPU eager，并把设备相关逻辑收敛到 backend。
```

## 阶段完成后的下一步

进入 Stage 02：设备后端抽象与 NPU eager。

Stage 01 的限制是所有逻辑都默认跑在普通 PyTorch/CPU 上。Stage 02 要解决的问题是：不要让业务代码到处写设备判断，而是通过 backend 抽象把 CPU 和 NPU 的差异集中管理。

但是进入 Stage 02 前，先确认 Stage 01 的这些事情已经稳定：

- 单 prompt 生成循环能跑。
- 每一步的 shape 你能解释。
- `Sequence` 的状态变化你能解释。
- 生成长度不会超过 `max_tokens`。
- 测试通过。

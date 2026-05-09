# Stage 03：Qwen3 模型结构与权重加载

## 这一阶段到底要做成什么

Stage 01 跑通了 tiny Transformer 的生成循环。Stage 02 把 CPU/NPU eager 后端抽出来了。Stage 03 要把 tiny 随机模型换成真实的 Qwen3 dense 小模型，例如 `Qwen/Qwen3-0.6B`。

这一阶段的最小闭环是：

```text
本地 Qwen3 模型目录
  -> 读取 config.json
  -> 构造 Qwen3ForCausalLM
  -> 从 safetensors 加载真实权重
  -> tokenizer 把 prompt 转成 input_ids
  -> backend 把 model/input_ids 放到 CPU 或 NPU
  -> 普通 PyTorch attention 执行一次 prefill
  -> 得到 logits
  -> 取最后一个 token 的 logits
  -> sampler 采样一个 next token
```

这一阶段不要求生成质量好，也不要求速度快。合格标准是：真实模型结构能建起来，真实权重能加载进去，一次 prefill forward 能跑通，`logits.shape` 正确，并且能采样出一个 token。

## 为什么 Stage 03 要单独做

真实模型一进来，问题会比 tiny model 多很多：

- tokenizer 可能没加载对。
- `config.json` 字段可能没解析全。
- `num_attention_heads` 和 `num_key_value_heads` 可能不同。
- `head_dim` 可能是 config 显式字段，不应该盲目猜。
- RoPE 可能写错。
- Q/K norm 可能漏掉。
- MLP 的 gate/up 顺序可能写反。
- safetensors 分片可能没加载全。
- 权重名可能和你的模块名对不上。
- dtype 和 device 可能不一致。

所以 Stage 03 的原则是：

```text
先用普通 PyTorch attention。
先只做完整 prefill。
先只支持一个 dense Qwen3 小模型。
先让结构和权重对上。
```

Stage 04 再做 KV cache，Stage 06 再接 Ascend fused attention。

## 本阶段学习目标

完成本阶段后，你应该能回答这些问题：

- `config.json` 里的字段如何决定模型结构。
- tokenizer 的 `input_ids` 如何进入真实模型。
- Qwen3 的 `embed_tokens`、decoder layers、final norm、`lm_head` 分别做什么。
- RMSNorm 和 Stage 01 的 LayerNorm 有什么区别。
- Qwen3 MLP 为什么是 gate/up/down 三个线性层。
- Qwen3 attention 里的 Q/K/V/O projection 怎么对应权重。
- `num_attention_heads` 和 `num_key_value_heads` 为什么可能不同。
- GQA 里 K/V head 如何 repeat 到 query head 数。
- RoPE 为什么作用在 Q/K 上，而不是 V 上。
- Q/K norm 为什么不能漏。
- safetensors 权重名如何对应到你的模块名。
- 为什么 prefill 后只取最后一个位置的 logits。

## 本阶段不做什么

这些内容先不要做：

- 不做 KV cache。
- 不做 paged attention。
- 不做 Ascend fused attention。
- 不调用 `torch_npu.npu_prompt_flash_attention`。
- 不调用 `torch_npu.npu_incre_flash_attention`。
- 不做 graph capture。
- 不做 tensor parallel。
- 不做 MoE 版 Qwen3。
- 不做 sliding window attention。
- 不做量化。
- 不做 OpenAI API server。
- 不追求输出速度。
- 不追求长文本生成质量。

如果 `config.json` 里出现本阶段不支持的特性，例如 MoE 或 sliding window，建议直接报一个清楚的错误，不要假装支持。

## 推荐先固定的模型

先只支持一个 dense 小模型：

```text
Qwen/Qwen3-0.6B
```

不要一开始写通用模型注册系统，也不要一开始兼容 Qwen3-MoE、Qwen2.5、Llama。先把一个模型跑通，再抽象。

推荐准备一个本地模型目录：

```text
/data/models/Qwen3-0.6B/
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  model.safetensors
```

如果是分片权重，可能是：

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

Stage 03 的 loader 至少要能处理：

- 单个 `model.safetensors`。
- 带 `model.safetensors.index.json` 的分片 safetensors。

## 目录和文件建议

项目总结构推荐最终放在：

```text
src/minivllm_ascend/
```

如果你现在按阶段学习，也可以继续用：

```text
src/stage03/
```

重点不是目录名字，而是模块边界。

### 推荐最终项目结构

```text
src/minivllm_ascend/
  models/
    __init__.py
    qwen3.py
  layers/
    __init__.py
    activation.py
    attention_torch.py
    embedding_head.py
    layernorm.py
    linear.py
    rotary_embedding.py
    sampler.py
  utils/
    __init__.py
    context.py
    loader.py
  backend/
    base.py
    torch_cpu.py
    torch_npu.py
  sampling_params.py
scripts/
  run_qwen3_torch_attention.py
tests/
  test_qwen3_shapes.py
  test_weight_loading.py
```

### 如果沿用 stage 目录

```text
src/stage03/
  models/
    qwen3.py
  layers/
    activation.py
    attention_torch.py
    embedding_head.py
    layernorm.py
    linear.py
    rotary_embedding.py
    sampler.py
  utils/
    context.py
    loader.py
  backend/
    base.py
    torch_cpu.py
    torch_npu.py
  sampling_params.py
scripts/
  run_qwen3_torch_attention.py
tests/
  test_qwen3_shapes.py
  test_weight_loading.py
```

Stage 03 可以复用 Stage 02 的 backend。不要在 Qwen3 模型代码里重新写：

```python
device = "cuda:0"
device = "npu:0"
```

模型只负责结构。设备由 backend 管。

## 推荐实现顺序

按这个顺序做，最容易定位问题：

1. 准备本地 Qwen3-0.6B 模型目录。
2. 写 `Qwen3Config.from_json`，先把 `config.json` 读出来。
3. 写 config 校验，遇到不支持的字段直接报错。
4. 写 `RMSNorm`，用小 tensor 测 shape。
5. 写 RoPE，单独测 q/k shape。
6. 写 MLP：gate/up/down + SiLU。
7. 写普通 PyTorch attention：Q/K/V/O projection、QK norm、RoPE、GQA、causal mask。
8. 写 decoder layer：attention residual + MLP residual。
9. 写 Qwen3Model：embedding、layers、final RMSNorm。
10. 写 Qwen3ForCausalLM：model + lm_head。
11. 先用 tiny config 随机权重跑 forward。
12. 写 safetensors loader。
13. 加载真实权重，打印 missing/unexpected keys。
14. 用 tokenizer 编码一个短 prompt，执行一次 prefill。
15. 取最后 token logits，采样一个 token。
16. 写 shape 测试和权重加载测试。

关键思路是：先验证结构，再验证权重。不要在结构没跑通时就开始处理复杂权重映射。

## 模块 1：`models/qwen3.py`

### 这个模块要干什么

`qwen3.py` 是 Stage 03 的核心文件。它负责定义 Qwen3 dense causal LM 的结构。

推荐拆成这些类：

```text
Qwen3Config
Qwen3MLP
Qwen3Attention
Qwen3DecoderLayer
Qwen3Model
Qwen3ForCausalLM
```

推荐调用链：

```text
Qwen3ForCausalLM.forward(input_ids, position_ids)
  -> Qwen3Model.forward(input_ids, position_ids)
    -> embed_tokens(input_ids)
    -> Qwen3DecoderLayer 0
      -> input_layernorm
      -> self_attn
      -> residual add
      -> post_attention_layernorm
      -> mlp
      -> residual add
    -> ...
    -> final norm
  -> lm_head
  -> logits
```

### `Qwen3Config` 应该保存什么

不要手写死模型参数。应该从本地 `config.json` 读取。

先支持这些字段：

```text
vocab_size
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
hidden_act
max_position_embeddings
rms_norm_eps
rope_theta
rope_scaling 或 rope_parameters
attention_bias
tie_word_embeddings
eos_token_id
bos_token_id
pad_token_id
```

字段含义：

```text
vocab_size:
  词表大小，决定 embedding 和 lm_head 的维度。

hidden_size:
  每个 token 的隐藏向量维度。

intermediate_size:
  MLP 中间层维度。

num_hidden_layers:
  decoder layer 层数。

num_attention_heads:
  query head 数。

num_key_value_heads:
  key/value head 数。小于 query head 数时就是 GQA。

head_dim:
  每个 head 的维度。优先使用 config 里的 head_dim。

rms_norm_eps:
  RMSNorm 的 eps。

rope_theta:
  RoPE 的 base。

attention_bias:
  q/k/v/o projection 是否有 bias。

tie_word_embeddings:
  lm_head 是否和 embed_tokens 共享权重。
```

### config 读取建议

```python
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    tie_word_embeddings: bool
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        data = json.loads(Path(path).read_text())
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data.get(
                "num_key_value_heads",
                data["num_attention_heads"],
            ),
            head_dim=data.get(
                "head_dim",
                data["hidden_size"] // data["num_attention_heads"],
            ),
            hidden_act=data.get("hidden_act", "silu"),
            max_position_embeddings=data["max_position_embeddings"],
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 10000.0),
            attention_bias=data.get("attention_bias", False),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            eos_token_id=data.get("eos_token_id"),
            bos_token_id=data.get("bos_token_id"),
            pad_token_id=data.get("pad_token_id"),
        )
```

### config 校验

读完 config 后立刻做校验：

```text
num_attention_heads % num_key_value_heads == 0
num_attention_heads * head_dim == hidden_size
hidden_act == "silu"
use_sliding_window 暂时为 False
rope_scaling / rope_parameters 暂时为空或 default
```

例如：

```python
if data.get("use_sliding_window", False):
    raise NotImplementedError("Stage 03 does not support sliding window attention")
```

新人阶段不要忽略不支持的配置。忽略配置字段会让模型看起来能跑，但语义可能已经错了。

## 模块 2：`layers/layernorm.py`

### 这个模块要干什么

Qwen3 使用 RMSNorm，不是 Stage 01 tiny model 里的 `LayerNorm`。

RMSNorm 大致做的是：

```text
x -> 按最后一维计算均方 -> rsqrt -> 乘回 x -> 乘可学习 weight
```

它没有“减均值”这一步。

### 推荐实现

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
```

输入输出：

```text
hidden_states: [batch, seq_len, hidden_size]
output: [batch, seq_len, hidden_size]
```

常见错误：

- 把 RMSNorm 写成 LayerNorm。
- 忘记 `weight` 参数。
- eps 用错。
- 在 float16/bfloat16 下直接算方差导致数值不稳。建议先转 float32 计算，再转回原 dtype。

权重名通常对应：

```text
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
```

## 模块 3：`layers/activation.py`

### 这个模块要干什么

Qwen3 的 MLP 是 gated MLP，常见形式是：

```text
down_proj( silu(gate_proj(x)) * up_proj(x) )
```

如果你做 gate/up 合并，可以写一个 `SiluAndMul`：

```python
class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

如果你先不合并 gate/up，也可以直接在 MLP 里写：

```python
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Stage 03 推荐先不合并。这样权重名最容易和 Hugging Face 对齐。

## 模块 4：`layers/linear.py`

### 这个模块要干什么

Stage 03 不做 tensor parallel，所以这里可以先封装普通 `nn.Linear`，也可以直接使用 `nn.Linear`。

如果封装，保持很薄：

```python
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

不要在这里写死 device：

```python
nn.Linear(..., device="npu:0")
```

正确方式是：

```text
先构造模型
再由 Stage 02 backend 统一 model.to(device)
```

## 模块 5：`layers/rotary_embedding.py`

### 这个模块要干什么

RoPE 是 rotary position embedding。它给 Q/K 注入位置信息。

注意：

```text
RoPE 作用在 Q 和 K 上。
RoPE 不作用在 V 上。
```

推荐 attention 内部使用这个 shape：

```text
q: [batch, seq_len, num_heads, head_dim]
k: [batch, seq_len, num_kv_heads, head_dim]
position_ids: [batch, seq_len]
```

RoPE 输出 shape 不变：

```text
q_rot: [batch, seq_len, num_heads, head_dim]
k_rot: [batch, seq_len, num_kv_heads, head_dim]
```

### 推荐实现思路

先写三个函数：

```text
build_inv_freq(head_dim, rope_theta)
build_cos_sin_cache(max_position, head_dim, rope_theta, device, dtype)
apply_rotary_pos_emb(q, k, position_ids, cos, sin)
```

核心逻辑：

```text
1. 根据 head_dim 和 rope_theta 生成 inv_freq。
2. 根据 position_ids 取对应位置的 cos/sin。
3. 把 q/k 的一半维度旋转。
4. 返回 q_rot/k_rot。
```

常见 `rotate_half` 写法：

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

RoPE 最容易出现“shape 对，但数值错”。最稳的办法是先做 shape 测试，再用 Hugging Face 模型做一个短 prompt 的 logits 对比。

### 新人需要注意

常见错误：

- RoPE 用在 hidden_states 上，而不是 Q/K 上。
- positions 从 1 开始，而不是从 0 开始。
- cos/sin shape broadcast 不对。
- head_dim 用错。
- 忽略 `rope_theta`。
- 只给 Q 用 RoPE，忘了 K。

Stage 03 只做 prefill，所以 position 可以先这样生成：

```python
position_ids = torch.arange(seq_len, device=input_ids.device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
```

Stage 04 做 KV cache 后，decode 的 position 会变成当前 token 的真实位置。

## 模块 6：`layers/attention_torch.py`

### 这个模块要干什么

实现 Qwen3 的普通 PyTorch attention。它不用 fused kernel，不用 KV cache，不用 paged cache。

它负责：

- Q/K/V projection。
- Q norm 和 K norm。
- RoPE。
- GQA 的 K/V repeat。
- causal attention。
- O projection。

### 推荐 attention 流程

```text
hidden_states: [B, T, hidden_size]
  -> q_proj -> q: [B, T, num_heads * head_dim]
  -> k_proj -> k: [B, T, num_kv_heads * head_dim]
  -> v_proj -> v: [B, T, num_kv_heads * head_dim]
  -> reshape q: [B, T, num_heads, head_dim]
  -> reshape k/v: [B, T, num_kv_heads, head_dim]
  -> q_norm(q)
  -> k_norm(k)
  -> apply RoPE(q, k, position_ids)
  -> repeat_kv(k/v) 到 num_heads
  -> transpose 到 [B, num_heads, T, head_dim]
  -> scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
  -> causal mask
  -> softmax
  -> attn @ v
  -> reshape 回 [B, T, hidden_size]
  -> o_proj
```

### 关键 shape

```text
B = batch
T = seq_len
H = num_attention_heads
H_kv = num_key_value_heads
D = head_dim
C = hidden_size = H * D

hidden_states: [B, T, C]
q: [B, T, H, D]
k: [B, T, H_kv, D]
v: [B, T, H_kv, D]
k_after_repeat: [B, T, H, D]
v_after_repeat: [B, T, H, D]
q_for_matmul: [B, H, T, D]
k_for_matmul: [B, H, T, D]
attention_scores: [B, H, T, T]
attention_output: [B, T, C]
```

### GQA 的 repeat_kv

如果：

```text
num_attention_heads = 16
num_key_value_heads = 8
```

每个 KV head 要服务 2 个 query head。

推荐写：

```python
def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    # input: [B, T, H_kv, D]
    if num_repeats == 1:
        return hidden_states
    batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch,
        seq_len,
        num_kv_heads,
        num_repeats,
        head_dim,
    )
    return hidden_states.reshape(batch, seq_len, num_kv_heads * num_repeats, head_dim)
```

### causal mask

Stage 03 先做完整 prefill，所以 mask 是 `[T, T]` 下三角：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
mask = mask.view(1, 1, seq_len, seq_len)
scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
```

如果 dtype 是 float32，可以用 `float("-inf")`。如果是 float16/bfloat16，用 `torch.finfo(scores.dtype).min` 更稳。

常见错误：

- 忘记 Q/K norm。
- GQA 没 repeat K/V，导致 matmul head 数不一致。
- mask 建在 CPU，hidden_states 在 NPU。
- softmax 在低精度下数值不稳。可以先把 scores 转 float32 做 softmax，再转回原 dtype。

## 模块 7：Qwen3 MLP

### 这个模块要干什么

Qwen3 MLP 是 SwiGLU 风格：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

推荐实现：

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

如果 config 里 `hidden_act` 不是 `silu`，Stage 03 可以先报错。

## 模块 8：`Qwen3DecoderLayer`

### 这个模块要干什么

一个 decoder layer 包含：

```text
input_layernorm
self_attn
residual add
post_attention_layernorm
mlp
residual add
```

推荐写成最容易理解的形式：

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

不要把 residual 放错位置。真实 Transformer 里很多“shape 正确但数值错”的 bug 都来自 residual 顺序错误。

## 模块 9：`Qwen3Model` 和 `Qwen3ForCausalLM`

### `Qwen3Model` 做什么

`Qwen3Model` 是不带 lm_head 的主体：

```text
embed_tokens
layers
norm
```

forward：

```text
input_ids -> hidden_states -> layers -> final norm -> hidden_states
```

### `Qwen3ForCausalLM` 做什么

`Qwen3ForCausalLM` 在 `Qwen3Model` 后面加 `lm_head`：

```text
hidden_states -> lm_head -> logits
```

输出：

```text
logits: [batch, seq_len, vocab_size]
```

推荐 forward：

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids)
        logits = self.lm_head(hidden_states)
        return logits
```

Stage 03 可以在模型里自动生成 `position_ids`：

```python
if position_ids is None:
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
```

后面 Stage 04 做 KV cache 时，decode 的 position 不再总是从 0 开始，所以要保留外部传入 `position_ids` 的能力。

## 模块 10：`utils/loader.py`

### 这个模块要干什么

`loader.py` 负责从 Hugging Face 格式的模型目录加载权重。

它应该做：

- 读取 `config.json`。
- 找到 safetensors 文件。
- 支持单文件 safetensors。
- 支持分片 safetensors index。
- 加载 state dict。
- 映射权重名。
- 调用 `model.load_state_dict(...)`。
- 打印 missing keys 和 unexpected keys。

### 依赖

Stage 03 建议安装：

```bash
pip install transformers safetensors
```

如果环境不能联网，就先确认已有环境：

```bash
python -c "import transformers, safetensors; print(transformers.__version__)"
```

### 单文件 safetensors

```python
from safetensors.torch import load_file

state_dict = load_file(model_path / "model.safetensors", device="cpu")
```

建议先加载到 CPU：

```text
权重文件 -> CPU state_dict -> model.load_state_dict -> backend.module_to_device(model)
```

不要一开始直接把 safetensors 加载到 NPU。CPU 路径更容易调试，也能避免 NPU 显存一开始就被撑爆。

### 分片 safetensors

如果有：

```text
model.safetensors.index.json
```

里面会记录每个权重在哪个分片文件。

推荐逻辑：

```python
def load_safetensors_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard in shard_files:
            state_dict.update(load_file(model_dir / shard, device="cpu"))
        return state_dict

    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return load_file(single_file, device="cpu")

    raise FileNotFoundError(f"no safetensors found in {model_dir}")
```

### 权重名映射

如果你的模块命名尽量对齐 Hugging Face，权重加载会简单很多。

建议模块名尽量使用：

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

这样第一版 loader 可以直接：

```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
```

然后打印：

```python
print("missing keys:", missing)
print("unexpected keys:", unexpected)
```

当 missing/unexpected 都解释清楚后，再改成严格加载。

### q/k/v 和 gate/up 要不要合并

原路线里提到 q/k/v 合并、gate/up 合并。这个可以做，但不建议作为第一步。

第一版最适合新人：

```text
q_proj, k_proj, v_proj 分开建。
gate_proj, up_proj 分开建。
HF 权重直接加载。
权重名最简单。
```

等模型能跑后，再做合并：

```text
qkv_proj = concat(q_proj.weight, k_proj.weight, v_proj.weight)
gate_up_proj = concat(gate_proj.weight, up_proj.weight)
```

这样每次只改变一个变量，容易排错。

## 模块 11：`utils/context.py`

### 这个模块要干什么

`context.py` 可以先很小。它的作用是集中保存一次 forward 需要的上下文信息。

Stage 03 可以定义：

```text
ForwardContext
  is_prefill: bool
  batch_size: int
  seq_len: int
  device: torch.device
  dtype: torch.dtype
```

也可以暂时不写这个文件。原路线里预留它，是为了后面 Stage 04/05 放：

- positions
- context_lens
- slot_mapping
- block_tables
- is_prefill

如果你现在还没感觉到它的必要性，可以先不抽象。Stage 03 更重要的是把模型和权重跑通。

## 模块 12：`scripts/run_qwen3_torch_attention.py`

### 这个脚本要干什么

这个脚本是 Stage 03 的手工验收入口。它只验证一件事：

```text
Qwen3 真实权重 + 普通 PyTorch attention 可以执行一次 prefill，并采样一个 token。
```

### 推荐命令

CPU 路径先跑：

```bash
PYTHONPATH=src python scripts/run_qwen3_torch_attention.py \
  --model-path /data/models/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --prompt "你好，介绍一下你自己" \
  --max-new-tokens 1
```

NPU eager 路径后跑：

```bash
PYTHONPATH=src python scripts/run_qwen3_torch_attention.py \
  --model-path /data/models/Qwen3-0.6B \
  --device npu \
  --dtype bfloat16 \
  --prompt "你好，介绍一下你自己" \
  --max-new-tokens 1
```

### 推荐参数

```text
--model-path:
  本地 Qwen3 模型目录。

--device:
  cpu 或 npu，复用 Stage 02 backend。

--dtype:
  float32、float16、bfloat16。初次调试建议 CPU float32。

--prompt:
  输入文本。

--max-new-tokens:
  Stage 03 可以先只支持 1。要生成多个 token 也可以，但每步都会重算完整上下文。

--compare-hf:
  可选。用 transformers AutoModelForCausalLM 做一次 logits 对比。
```

### 脚本执行流程

```text
1. parse args。
2. get_backend(args.device)。
3. AutoTokenizer.from_pretrained(args.model_path)。
4. Qwen3Config.from_json(model_path / "config.json")。
5. 构造 Qwen3ForCausalLM(config)。
6. load_qwen3_weights(model, model_path)。
7. model.to(dtype)。
8. backend.module_to_device(model)。
9. tokenizer(prompt, return_tensors="pt")。
10. backend.to_device(input_ids)。
11. logits = model(input_ids)。
12. last_token_logits = logits[:, -1, :]。
13. next_token = sampler.sample(last_token_logits)。
14. tokenizer.decode(next_token)。
15. 打印 shape、device、next token。
```

期望输出类似：

```text
model_path=/data/models/Qwen3-0.6B
backend=cpu
dtype=float32
input_ids.shape=torch.Size([1, 12])
logits.shape=torch.Size([1, 12, 151936])
last_token_logits.shape=torch.Size([1, 151936])
next_token_id=...
next_token_text=...
```

具体 token 不要求固定。

### tokenizer 注意事项

Stage 03 可以直接使用：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"]
```

先不要做复杂 chat template。你可以直接传普通文本。

## 模块 13：`tests/test_qwen3_shapes.py`

### 这个测试要验证什么

shape 测试不应该依赖真实 Qwen3-0.6B 权重，否则测试太慢、太吃内存。

推荐用一个 tiny Qwen3Config：

```text
vocab_size = 128
hidden_size = 32
intermediate_size = 64
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 2
head_dim = 8
max_position_embeddings = 64
```

然后验证：

```text
input_ids: [2, 5]
logits: [2, 5, 128]
```

推荐测试点：

- RMSNorm 输入输出 shape 一致。
- RoPE 后 q/k shape 一致。
- attention 输出 `[B, T, hidden_size]`。
- MLP 输出 `[B, T, hidden_size]`。
- Qwen3ForCausalLM 输出 `[B, T, vocab_size]`。
- GQA 下 `num_attention_heads != num_key_value_heads` 也能跑。

## 模块 14：`tests/test_weight_loading.py`

### 这个测试要验证什么

权重加载测试分两层：

```text
轻量单元测试:
  用假的 tiny state_dict 或临时 safetensors 文件，验证 loader 能加载。

真实模型集成测试:
  如果设置了 QWEN3_MODEL_PATH，就加载真实 Qwen3 权重跑一次。
  如果没设置，就 skip。
```

真实模型测试默认 skip，是为了避免没有模型文件的机器无法跑基础测试。

示例：

```python
def test_load_real_qwen3_if_available():
    model_path = os.environ.get("QWEN3_MODEL_PATH")
    if not model_path:
        pytest.skip("QWEN3_MODEL_PATH is not set")

    config = Qwen3Config.from_json(Path(model_path) / "config.json")
    model = Qwen3ForCausalLM(config)
    load_qwen3_weights(model, model_path)
```

## 关键 shape 总结

### 整体 forward

```text
input_ids: [batch, seq_len]
position_ids: [batch, seq_len]
hidden_states: [batch, seq_len, hidden_size]
logits: [batch, seq_len, vocab_size]
last_token_logits: [batch, vocab_size]
next_token_ids: [batch]
```

### attention

```text
q: [batch, seq_len, num_attention_heads, head_dim]
k: [batch, seq_len, num_key_value_heads, head_dim]
v: [batch, seq_len, num_key_value_heads, head_dim]

k_after_repeat: [batch, seq_len, num_attention_heads, head_dim]
v_after_repeat: [batch, seq_len, num_attention_heads, head_dim]

q_for_matmul: [batch, num_attention_heads, seq_len, head_dim]
k_for_matmul: [batch, num_attention_heads, seq_len, head_dim]
v_for_matmul: [batch, num_attention_heads, seq_len, head_dim]

attention_scores: [batch, num_attention_heads, seq_len, seq_len]
attention_output: [batch, seq_len, hidden_size]
```

### MLP

```text
x: [batch, seq_len, hidden_size]
gate: [batch, seq_len, intermediate_size]
up: [batch, seq_len, intermediate_size]
silu(gate) * up: [batch, seq_len, intermediate_size]
down_proj output: [batch, seq_len, hidden_size]
```

## prefill 在 Stage 03 怎么理解

Stage 03 仍然只做无 KV cache 的完整 prefill：

```text
输入整个 prompt
每一层 attention 都看完整 prompt 的历史位置
输出每个位置的 logits
只取最后一个位置 logits 来预测下一个 token
```

例如 prompt 长度是 12：

```text
input_ids: [1, 12]
logits: [1, 12, vocab_size]
last_token_logits = logits[:, -1, :]
next_token = sampler(last_token_logits)
```

如果你在 Stage 03 生成多个 token，每一步仍然会重算完整上下文：

```text
step 0: 输入 prompt
step 1: 输入 prompt + token_0
step 2: 输入 prompt + token_0 + token_1
```

这很慢，但符合 Stage 03 的目标。Stage 04 才会引入 KV cache，让 decode 每步只输入最新 token。

## 和 Hugging Face 做对比

如果你想确认模型结构是否真的对，可以做一个可选对比：

```text
同一个 model_path
同一个 prompt
同一个 dtype
同一个 input_ids
你的 Qwen3ForCausalLM 输出 logits
HF AutoModelForCausalLM 输出 logits
比较 logits[:, -1, :100] 或 top-k token
```

建议先在 CPU float32 下比较。NPU/bfloat16 下允许误差更大。

如果差距很大，优先检查：

- RoPE 实现。
- Q/K norm。
- residual 顺序。
- MLP gate/up 顺序。
- 权重名映射。

## 任务清单

- 准备本地 `Qwen3-0.6B` 模型目录。
- 安装或确认 `transformers`。
- 安装或确认 `safetensors`。
- 实现 `Qwen3Config.from_json`。
- 实现 config 校验。
- 实现 `RMSNorm`。
- 实现 MLP 内联 SiLU gate，或实现 `SiluAndMul`。
- 实现 RoPE。
- 实现 `repeat_kv`。
- 实现 Qwen3 attention 的 q/k/v/o projection。
- 实现 Q/K norm。
- 实现 GQA。
- 实现 causal mask。
- 实现 Qwen3 MLP。
- 实现 Qwen3 decoder layer。
- 实现 Qwen3Model。
- 实现 Qwen3ForCausalLM。
- 用 tiny config 跑随机权重 forward。
- 实现 safetensors loader。
- 支持单文件 safetensors。
- 支持分片 safetensors index。
- 打印 missing keys 和 unexpected keys。
- 加载真实 Qwen3 权重。
- 写 `scripts/run_qwen3_torch_attention.py`。
- 用 tokenizer 编码短 prompt。
- 执行一次 prefill。
- 打印 logits shape。
- 采样一个 next token。
- 写 `tests/test_qwen3_shapes.py`。
- 写 `tests/test_weight_loading.py`。
- 写 Stage 03 学习记录。

## 推荐调试顺序

如果跑不通，按这个顺序查：

1. 先确认 `config.json` 能读取，并打印关键字段。
2. 用 tiny config 构造模型，先不要加载真实权重。
3. 单独测 RMSNorm shape。
4. 单独测 RoPE shape。
5. 单独测 attention shape，尤其是 GQA。
6. 单独测 MLP shape。
7. 跑完整 Qwen3ForCausalLM 随机权重 forward。
8. 加载 safetensors 后打印 missing/unexpected keys。
9. 如果 missing 很多，先检查模块命名是否对齐 HF。
10. 如果 shape mismatch，检查 `head_dim`、head 数、intermediate_size。
11. 如果 logits 数值和 HF 差很远，检查 RoPE、Q/K norm、MLP gate/up 顺序。
12. CPU float32 跑通后，再切 NPU bfloat16。

不要一边改结构，一边改 loader，一边改 NPU。一次只排一个变量。

## 常见坑

- 把 Qwen3 当成 Stage 01 tiny Transformer，只写普通 LayerNorm。
- 忘记 Q/K norm。
- `num_key_value_heads` 当成 `num_attention_heads`。
- GQA 没 repeat K/V。
- `head_dim` 直接写成 `hidden_size // num_heads`，但没有校验 config。
- RoPE 维度旋转方式和参考实现不一致。
- positions 从 1 开始。
- MLP 里写成 `silu(up) * gate`，gate/up 顺序错。
- residual add 顺序错。
- `lm_head.weight` 漏加载。
- `tie_word_embeddings` 没处理。
- safetensors 分片只加载了第一个 shard。
- `strict=False` 后不看 missing/unexpected keys。
- tokenizer 输出在 CPU，模型在 NPU。
- dtype 不一致。
- 直接在模型文件里写死 `device="npu:0"`。
- NPU 显存不够时误以为模型结构错。

## 验收标准

### 1. tiny config shape 测试通过

```bash
PYTHONPATH=src pytest tests/test_qwen3_shapes.py
```

至少验证：

- RMSNorm shape。
- attention shape。
- GQA shape。
- Qwen3ForCausalLM logits shape。

### 2. 权重加载测试通过

```bash
PYTHONPATH=src pytest tests/test_weight_loading.py
```

轻量测试必须通过。真实模型测试如果没有设置 `QWEN3_MODEL_PATH`，应该 skip。

### 3. Qwen3 prefill demo 能跑

```bash
PYTHONPATH=src python scripts/run_qwen3_torch_attention.py \
  --model-path /data/models/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --prompt "你好" \
  --max-new-tokens 1
```

应该看到：

- config 关键字段。
- tokenizer 输出的 `input_ids.shape`。
- 权重加载 missing/unexpected keys。
- `logits.shape`。
- `last_token_logits.shape`。
- 采样得到的 next token id。

### 4. NPU eager 可选验收

如果 Stage 02 NPU backend 已经稳定，可以跑：

```bash
PYTHONPATH=src python scripts/run_qwen3_torch_attention.py \
  --model-path /data/models/Qwen3-0.6B \
  --device npu \
  --dtype bfloat16 \
  --prompt "你好" \
  --max-new-tokens 1
```

如果 NPU 显存不够，可以先只把 Stage 03 CPU 路径作为必过验收，并在学习记录里写清楚 NPU 阻塞原因。

## 学习记录建议

完成后，在 Stage 03 学习记录里写这些内容：

```text
本阶段目标:
  用真实 Qwen3 dense 模型结构替换 tiny model，并加载 safetensors 权重跑一次 prefill。

模型信息:
  model_path:
  vocab_size:
  hidden_size:
  intermediate_size:
  num_hidden_layers:
  num_attention_heads:
  num_key_value_heads:
  head_dim:
  rms_norm_eps:
  rope_theta:
  dtype:
  device:

当前调用链:
  scripts/run_qwen3_torch_attention.py
    -> AutoTokenizer.from_pretrained
    -> Qwen3Config.from_json
    -> Qwen3ForCausalLM
    -> load_qwen3_weights
    -> backend.module_to_device
    -> model(input_ids)
    -> Qwen3Attention
    -> Qwen3MLP
    -> lm_head
    -> sampler

关键 shape:
  input_ids:
  position_ids:
  hidden_states:
  q:
  k:
  v:
  attention_scores:
  attention_output:
  logits:
  last_token_logits:

权重加载结果:
  missing keys:
  unexpected keys:
  shape mismatch:

当前限制:
  没有 KV cache。
  decode 仍然重算完整上下文。
  attention 仍然是普通 PyTorch 实现。
  没有 paged attention。
  没有 Ascend fused attention。
  没有 tensor parallel。

遇到的问题:
  记录 RoPE、Q/K norm、GQA、safetensors、dtype、device 相关问题。

下一阶段为什么要做:
  Stage 04 要实现连续 KV cache，让 decode 不再每一步重算整个 prompt。
```

## 阶段完成后的下一步

进入 Stage 04：连续 KV cache。

Stage 03 的限制是：虽然已经能跑真实 Qwen3，但每生成一个 token 都要把完整上下文重新送进模型。

Stage 04 要解决的问题是：

```text
prefill:
  计算整个 prompt 的 K/V，并写入 cache。

decode:
  每步只输入最新 token。
  attention 读取历史 K/V cache。
```

进入 Stage 04 前，先确认：

- Qwen3 config 能正确读取。
- tiny config shape 测试通过。
- 真实 safetensors 能加载。
- prefill logits shape 正确。
- 你能解释 Qwen3 attention 的 Q/K/V/GQA/RoPE/QK norm。
- 你能解释为什么现在 decode 很慢，以及 KV cache 要解决什么。

## 参考资料

- Hugging Face Transformers Qwen3 文档：https://huggingface.co/docs/transformers/model_doc/qwen3
- 本阶段实际实现必须以本地模型目录里的 `config.json` 和 safetensors 权重名为准。

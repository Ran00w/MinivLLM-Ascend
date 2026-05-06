from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    max_model_length: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
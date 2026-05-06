import torch
import torch.nn as nn

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def sample(self, logits):
        if logits.ndim != 2:
            raise ValueError("logit must have the shape of [batch,vocab_size]")
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = torch.argmax(probs, dim=-1)
        return sample_tokens
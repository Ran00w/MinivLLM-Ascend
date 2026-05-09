import torch
import torch.nn as nn

class Rotary_embedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(0, max_seq_len).float()

        freq = torch.einsum("i j->ij", t, inv_freq)
        emb = torch.cat([freq, freq], dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, x):
        
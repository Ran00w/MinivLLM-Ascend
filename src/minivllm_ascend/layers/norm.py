import torch
import torch.nn as nn

class norm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps)

    def forward(self, x):
        return self.norm(x)
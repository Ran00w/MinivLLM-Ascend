import torch
import torch.nn as nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = torch.chunk(x, 2 ,dim=-1)
        return F.silu(gate) * up
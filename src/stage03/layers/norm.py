import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps     

    def forward(self, x):
        input_type = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(input_type)
    

if __name__ == '__main__':
    layer = RMSNorm(1024)
    input = torch.randn(1024,dtype=torch.float32)
    print(input)
    output = layer(input)
    print(output)
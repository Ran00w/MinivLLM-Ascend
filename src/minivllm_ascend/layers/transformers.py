import torch
import torch.nn as nn
import torch.functional as F
from attention_torch import attention
from norm import norm

device = "cuda:0"

class TinyTransformerConfig:
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size

class decoder_block(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = attention(hidden_size=hidden_size, num_head=num_heads)
        self.layernorm1 = norm(hidden_size)
        self.layernorm2 = norm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, intermediate_size, device=device)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, device=device)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        attn_score = self.attn(self.layernorm1(x)) + x
        ffn_ans = self.linear2(self.relu(self.linear1(self.layernorm2(attn_score)))) + x
        return ffn_ans



class transformers(nn.Module):
    def __init__(self, vocab_size = 10000, hidden_size = 64, num_layers = 12, 
                 num_heads = 4, intermediate_size = 256, 
                 max_position_embeddings = 1024):
        super().__init__()
        config = TinyTransformerConfig(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings)
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device)
        self.decoder_blocks = nn.ModuleList([decoder_block(hidden_size, num_heads, intermediate_size)
                                             for _ in range(num_layers)
                                             ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, device=device)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.decoder_blocks:
            x = block(x)

        return self.lm_head(x)
    
if __name__ == '__main__':

    layer = transformers()

    input = torch.randint(0, 10000, (64, 512), device=device)
    input = input.long()
    
    output = layer(input)
        

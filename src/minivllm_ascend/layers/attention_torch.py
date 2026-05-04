import torch
import torch.nn as nn

# easy casuallm
# input: [batch_size, seq_len, hidden_size] ->
# q: [batch_size, num_head, seq_len, hidden_size//num_head]
# k: [batch_size, num_head, seq_len, hidden_size//num_head]
# v: [batch_size, num_head, seq_len, hidden_size//num_head]
# score: [batch_size, num_head, seq_len, seq_len]
# output: [batch_size, seq_len, hidden_size]
device = "cuda:0"

class attention(nn.Module):
    def __init__(self, hidden_size, num_head):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.k_proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.v_proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.o_proj = nn.Linear(hidden_size, hidden_size, device=device)

#        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
#        self.mask = mask.view(1, 1, seq_len, seq_len)

        self.num_head = num_head

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # [batch_size, seq_len, hidden_size] -> [batch_size, num_head, seq_len, hidden_size//num_head]
        batch_size, seq_len, hidden_size = Q.shape
        assert hidden_size % self.num_head == 0, "num_head should have the right size"
        head_size = hidden_size // self.num_head 
        Q = Q.view(batch_size, seq_len, self.num_head, head_size)
        Q = torch.einsum('b s n h -> b n s h', Q)
        K = K.view(batch_size, seq_len, self.num_head, head_size)
        K = torch.einsum('b s n h -> b n s h', K)
        K = torch.einsum('b n s h -> b n h s', K)
        V = V.view(batch_size, seq_len, self.num_head, head_size)
        V = torch.einsum('b s n h -> b n s h', V)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.view(1, 1, seq_len, seq_len)

        attention_score = torch.matmul(Q, K) / (head_size ** 0.5)
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attention_score, dim=-1)
        attn = attn @ V

        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        output = self.o_proj(attn)

        return output


if __name__ == '__main__':

    layer = attention(1024,8)
    input = torch.rand(32, 1024, 1024, device=device)
    output = layer(input)

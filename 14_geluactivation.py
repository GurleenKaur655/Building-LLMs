import torch 
import torch.nn as nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, 
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  ## Contraction
        )

    def forward(self, x):
        return self.layers(x)


print(GPT_CONFIG_124M["emb_dim"])
# 768

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)
# torch.Size([2, 3, 768])
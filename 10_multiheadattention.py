import torch.nn as nn 
import torch
import importlib
casualattention = importlib.import_module("09_casualattention")
CasualAttention = casualattention.CasualAttention
d_in, d_out = casualattention.d_in, casualattention.d_out
batch = casualattention.batch

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3,2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
#we implemented a MultiHeadAttentionWrapper that combined multiple single-head attention modules. However, note that these are processed sequentially via [head(x) for head in self.heads] in the forward method. can improve by processing the heads in parallel
# tensor([[[-0.5337, -0.1051,  0.5085,  0.3508],
#          [-0.5323, -0.1080,  0.5084,  0.3508],
#          [-0.5323, -0.1079,  0.5084,  0.3506],
#          [-0.5297, -0.1076,  0.5074,  0.3471],
#          [-0.5311, -0.1066,  0.5076,  0.3446],
#          [-0.5299, -0.1081,  0.5077,  0.3493]],

#         [[-0.5337, -0.1051,  0.5085,  0.3508],
#          [-0.5323, -0.1080,  0.5084,  0.3508],
#          [-0.5323, -0.1079,  0.5084,  0.3506],
#          [-0.5297, -0.1076,  0.5074,  0.3471],
#          [-0.5311, -0.1066,  0.5076,  0.3446],
#          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)

#gpt 3 has 96 attention heads in the largest version which has 175 billion parameters
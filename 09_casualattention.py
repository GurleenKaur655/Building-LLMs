import importlib
import torch.nn as nn
import torch
selfattention = importlib.import_module("08_selfattention")
SelfAttention_v2 = selfattention.SelfAttention_v2
d_in, d_out = selfattention.d_in, selfattention.d_out

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)3
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs) #A
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
#         [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
#         [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<SoftmaxBackward0>)
#this is the unmasked attention weights

context_length = attn_scores.shape[0] #for rows
#we use the PyTorch's tril function to mask the values above the diagonal are zero and triu is to do the opposite
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(torch.ones(context_length, context_length))
# tensor([[1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1.]])
print(mask_simple)
# tensor([[1., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1.]])

# multiply this mask with the attention weights to zero out the values above the diagonal 

masked_simple = attn_weights * mask_simple
print(masked_simple)
# tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<MulBackward0>)

# renormalize the attention weights to sum up to 1 again in each row

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# For a 2D tensor, dim=1 and dim=-1 are the same thing.
# masked_simple has shape (6, 6), so it has only two dimensions:
# dim=0 (or dim=-2) runs down the rows
# dim=1 (or dim=-1) runs across the columns

# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<DivBackward0>)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
#diagonal says where the triangle starts, with = 1 it starts one step above the main diagonal
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked) #masked_fill lay the map of mask over the attn_scores which is the same 4 * 4 shape
# tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
#         [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
#         [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
#         [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
#         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
#         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
#        grad_fn=<MaskedFillBackward0>)

# now we apply the softmax 
attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=1)
print(attn_weights)

#in softmax the exponential of -inf is 0

# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#        grad_fn=<SoftmaxBackward0>)

#in this example we use, the dropout of 50%, which means masking out the half of the attention weights 

#in GPT its usually lower dropout rate such as 0.1, 0.2

#random example first
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))
# tensor([[2., 2., 0., 2., 2., 0.],
#         [0., 0., 0., 2., 0., 2.],
#         [2., 2., 2., 2., 0., 2.],
#         [0., 2., 2., 0., 0., 2.],
#         [0., 2., 0., 2., 0., 2.],
#         [0., 2., 2., 2., 2., 0.]])

#to compensate for the reduction in active elements the values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 = 2
#this scaling is crucial to maintain the overall balance of the attention weights, ensuring that the average influence of the attention mechanism remains consistent during both the training and inference phases.
# The scale factor is 1 / (1 − p), where p is the dropout rate (the fraction turned off), not the fraction kept. so if the drop out is 0.2 then its 1/0.8 for it 


torch.manual_seed(123)
print(dropout(attn_weights))
# tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
#         [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
#        grad_fn=<MulBackward0>)

#we need to ensure the code can handle batched consisting of more than one input, which will ensure that the casual attention supports the batch outputs produced by the data loader we implemented earlier

# 2 inputs with 6 tokens each, and each token has embedding dimension 3
#we are taking the same 6 tokens as another batch as an example
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
# torch.Size([2, 6, 3])

#results in 3D tensor consisting of 2 input texts with 6 tokens each, where each token is a 3-dimensional embedding vector

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value =  nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape #New batch dimension b [b, 6, 3]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) #changes transpose
        # .transpose(1, 2) swaps dimension 1 and dimension 2 and leaves dimension 0 (the batch) alone. .T reverses all the dimensions
        attn_scores.masked_fill( #New, _ops are in-place 
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # :num_tokens to account for the last batch in a text which might not have the same context length as the other batches
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) #New

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
# context_vecs.shape: torch.Size([2, 6, 2])
print(context_vecs)
# tensor([[[-0.5337, -0.1051],
#          [-0.5323, -0.1080],
#          [-0.5323, -0.1079],
#          [-0.5297, -0.1076],
#          [-0.5311, -0.1066],
#          [-0.5299, -0.1081]],

#         [[-0.5337, -0.1051],
#          [-0.5323, -0.1080],
#          [-0.5323, -0.1079],
#          [-0.5297, -0.1076],
#          [-0.5311, -0.1066],
#          [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)
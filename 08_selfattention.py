import torch
import torch.nn as nn
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] #A 
d_in = inputs.shape[1] #B 
#d_in is basically dimension in: how many numbers each word's vector has going into the attention layer. Here thats 3
d_out = 2 #C #d_out is dimension out: how many numbers each word's query, key, value has coming out. Here its 2
#in GPT models the input and output dimensions are usually the same for illustraton purposes we are using 2 for output

torch.manual_seed(123) # same values come 
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #makes a tensor as a learnable parameter of a model, one of the numbers model updagtes during training, grad = false, turns it off for this model for now, a plain tensor would be ignored and never learn
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query)
print(W_key)
print(W_value)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query

print("keys.shape:", keys.shape)
#keys.shape: torch.Size([6, 2]) we have changes the tokens from the 3D embedding space to the 2D space

attn_scores_2 = query_2 @ keys.T # all attention score for the given query, for a single attention score, that certain query with transpose of the keys

print(attn_scores_2)

attn_scores = queries @ keys.T # omega
print(attn_scores)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
print(d_k)

context_vec2 = attn_weights_2 @ values
print(context_vec2)

#creating self attention class 

class SelfAttention_v1(nn.Module):
    #Module is the fundamental building block of PyTorch models, which provides necessary functionalities for model layer creation and management.
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        #we will change nn.Parameter to nn.Linear because it has more stable optimization
        self.W_key = nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = nn.Parameter(torch.rand(d_in,d_out))
    #init method initializes trainable weight matrices(W_query, key and value), each tranforming the input dimension d_in to an d_out
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
    #x is the input matrix here 

        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
# tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)

class SelfAttention_v2(nn.Module):
    #Module is the fundamental building block of PyTorch models, which provides necessary functionalities for model layer creation and management.
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        #we will change nn.Parameter to nn.Linear because it has more stable optimization
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    #init method initializes trainable weight matrices(W_query, key and value), each tranforming the input dimension d_in to an d_out
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
    #x is the input matrix here 

        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# tensor([[-0.0739,  0.0713],
#         [-0.0748,  0.0703],
#         [-0.0749,  0.0702],
#         [-0.0760,  0.0685],
#         [-0.0763,  0.0679],
#         [-0.0754,  0.0693]], grad_fn=<MmBackward0>) different from the above found tensor, its more optimised
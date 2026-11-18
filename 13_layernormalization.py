import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2,5) # random numbers with mean 0 and variance 1

layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#        grad_fn=<ReluBackward0>)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Mean:
#  tensor([[0.1324],
#         [0.2170]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[0.0231],
#         [0.0398]], grad_fn=<VarBackward0>)

out_norm = (out - mean)/torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# Normalized layer outputs:
#  tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
#         [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
#        grad_fn=<DivBackward0>)
# Mean:
#  tensor([[-5.9605e-08], practically almost 0
#         [ 1.9868e-08]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

#can turn off the scientific mode 

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# Mean:
#  tensor([[-0.0000],
#         [ 0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # tensor with scalar value 1 and same size as emb_dim
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # tensor with scalar values 0 and same size as the emb_dim

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
# the variable eps is a small constant (epsilon) adde dto the variance to prevent division by zero during normalization

# the scale and shift are two trainable parameters (of the same dimension as the input) that the LLM automatically adjust during the training if it is determined that doing so would imporve the model's performance on its training task(trainable means we dont define their values at start)

#this allows model to learn the appropriate scaling and shifting that best suit that data it is processing

# if unbiased = True, in variance calculation we divide by the number of inputs in the variance formula, this approach does not apply the Bessel's correction, which typically uses n-1 instead of n in the denominatior to adjust for bias in sample variance estimation, this decision results in so-called biased estimate of the variance, for large-scale LLMs where the embedding dimension is large the difference between n and n-1 is practically neglible.
# we use because of the orginal GPT - 2 model 

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=1, unbiased=False, keepdim=True)
print(mean)
print(var)

# tensor([[-0.0000],
#         [ 0.0000]], grad_fn=<MeanBackward1>)
# tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)
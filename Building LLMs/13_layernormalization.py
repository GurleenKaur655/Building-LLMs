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
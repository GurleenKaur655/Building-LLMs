import torch

input_ids = torch.tensor([2,3,5,1])
# will use size of vocab 6, and want to create embeddings of size 3 dimensions

vocab_size = 6
output_dim = 3  

torch.manual_seed(123)
# Sets the starting point (seed) of PyTorch's random number generator, so "random" numbers come out the same on every run. When creating an embedding layer, this means each token ID's row in the embedding weight matrix starts with the same random values every time. It only affects the starting values, which change during training. It's used for reproducible results, easier debugging, and matching the course output. The seed number itself (123, 42…) has no special meaning, and the order of random operations matters.
embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # embedding layer is a lookup table that maps each token ID to a dense vector representation of the token in a continuous space. It stores the embeddings of a fixed dictionary and size.
print (embedding_layer.weight) # this will give the embeddings of the input_ids
# Parameter containing:
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True) all are random small values

print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
# look up table beacuse it just look for that token id and gives u that vector from that embedding weight matrix

print(embedding_layer(input_ids))
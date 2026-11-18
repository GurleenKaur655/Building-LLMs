import torch
import importlib
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()
data_loader = importlib.import_module("04_data_loader")
create_dataloader_v1 = data_loader.create_dataloader_v1

vocab_size = 50257 # the input nodes
output_dim = 256 # the output dimensions

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#using the token_emdedding_layer above, if we sample data from the data loader, we embed each token in each batch, and if batch size of 8 with four tokens each, the result will be an 8 * 4 * 256 tensor

max_length = 4 
dataloader = create_dataloader_v1(raw_txt, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print(inputs)
print(inputs.shape)

# tensor([[10970, 33310,    35, 18379],
#         [  198, 15749, 40417,   198],
#         [  198,    40,   550,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  198,   198, 11274,  5891],
#         [ 1576,   438,   568,   340]])
# torch.Size([8, 4]) its just the first batch of 8 sequences of 4 tokens each, the first dimension is the batch size and the second dimension is the sequence length
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) #torch.Size([8, 4, 256])

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # create the sequence of numbers 0,1... up to the maximum length-1
print(pos_embeddings.shape)
#torch.Size([4, 256])

input_embeddings = token_embeddings + pos_embeddings 
print(input_embeddings.shape)
#torch.Size([8, 4, 256])

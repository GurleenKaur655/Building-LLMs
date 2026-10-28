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

#Step 1: use a placeholder for transformer block
#Step 2 : use a placeholder for LayerNorm

# inherits from the Pytorch's base class so it can hold the trainable weights
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() #__init__: building the layers
        #runs when we create the model, cfg is a config dictionary such as {"vocab_size": 50257, "emb_dim":768, ...}
        #super().__init__(): calls the nn.Module setup, required so PyTorch can track all the layers and parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # makes a embedding matrix or like a table shape is (vocab_size, emb_dim) which is (50257, 768)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        #for the positional embedding as done its across the context length and added to the token embedding, so (1024, 768) and transformers dont know word order on their oen, so this tells the model which token is first, second and so on
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # prevents overfitting

        #use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        #use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx): # how data flows through
        batch_size, seq_len = in_idx.shape
        #in_idx is a batch of token IDs with shape(batch_size, seq_len), gets the dimeansions, eg 2 sentences of 4 tokens each gives batch_size=2, seq_len=4
        tok_embeds = self.tok_emb(in_idx) #looks up the vector for each token. the sahpe becomes (batch_size, serq_len, emb_dim) -> (2,4, 768)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) #torch.arange(seq_len) gives [0,1,2,3], the positions, it looks up a vector for each position, giving shape (seq_len, emb_dim), -> 4, 768 and device = in_idx.device keeps it on the same CPU or GPU as the input
        x = tok_embeds + pos_embeds#(2,4,768) + (4,768): same position vectors are added to every sentence in the batch
        x = self.drop_emb(x) # applies dropout
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
            #A simple placeholder

    def forward(self, x):
        # this block does nothing and just returns its input
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
            #the parameter here are just to mimic LayerNorm interface

    def forward(self, x):
            #this layer does nothign and just returns its input.
        return x
        # we made this as a separate class because normalization layer is used at multiple areas within the transformer and outside the transformer as well
        # There’s a second reason to make it a class besides reuse: each LayerNorm has its own learnable parameters (scale and shift). Because it’s an nn.Module, every place you create one gets its own trainable weights. A plain function couldn’t hold those.

#Step 1: Tokenization

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
# tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Outputs shape:", logits.shape)
print(logits)

# Outputs shape: torch.Size([2, 4, 50257])
# tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
#          [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
#          [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
#          [ 0.0139,  1.6754, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],

#         [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
#          [-0.7860,  0.5581, -0.0610,  ...,  0.4835, -0.0077,  1.6621],
#          [ 0.3567,  1.2698, -0.6398,  ..., -0.0162, -0.1296,  0.3717],
#          [-0.2407, -0.7349, -0.5102,  ...,  2.0057, -0.3694,  0.1814]]],
#        grad_fn=<UnsafeViewBackward0>)


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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0),  \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #reduce the projection dim to match the desrired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # linear layer to combine the head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) #shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        #we implicitly split the matrix by adding a 'num_heads' dimension
        #unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        #transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #compute the sclaed dot product attention (aka self-attnetion) with a casual mask 
        attn_scores = queries @ keys.transpose(2,3) # dot product for each head

        #original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        #use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)

        #combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
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
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
       return 0.5 * x * (1 + torch.tanh(
    torch.sqrt(torch.tensor(2.0 / torch.pi)) *
    (x + 0.044715 * torch.pow(x, 3))
))
    
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



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #shortcut connection for attention block
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x) # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # add the original input back 

        #shortcut connection for the feed forward block 
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x) # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # add the original input back
        return x

#Step 1: use a placeholder for transformer block
#Step 2 : use a placeholder for LayerNorm

# inherits from the Pytorch's base class so it can hold the trainable weights
class GPTModel(nn.Module):
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
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        #use a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # out_head is the linear layer (neural network layer) that takes the final hidden states from the transformer and projects them to the size of the vocabulary, so we can get a probability distribution over the vocabulary for each token position in the input sequence. This is used for predicting the next token in a sequence.
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
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input shape:", batch)
print("Outputs shape:", out.shape)
print(out)
# Input shape: tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])
# Outputs shape: torch.Size([2, 4, 50257])

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

#using numel() method, short for "number of elements", we can collect the total number of parameters in the model's parameter tensors.

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters in the model:", total_params)
# Total parameters in the model: 163009536
# why 163 million? and not 124 million?
# the reason is a concept called weight tying that is used in the original GPT-2 architecture, which means that the original GPT-2 architecture is reusing the weights from the token embedding layer in its output layer

print("Token embedding weights shape:", model.tok_emb.weight.shape)
# Token embedding weights shape: torch.Size([50257, 768])
print("Output head weights shape:", model.out_head.weight.shape)
# Output head weights shape: torch.Size([50257, 768])

#can remove the output parameters to check the parameters, as we are using all unique parameters in the model, so the total parameters should be 163 million, but the GPT-2 reuses the parameters used in the token embedding layer in the output layer, so the total parameters should be 124 million, which is the original GPT-2 model size.
total_params_without_output = sum(p.numel() for p in model.parameters() if p is not model.out_head.weight)
print("Total parameters without considering weight tying:", total_params_without_output)

# Total parameters without output head: 124412160


# the weight tying reduce the overall memory footprint and computational complexity of the model. However, in my experience using separate token embeddings and output layer results in better training and model performance, hence we are using separater layers in our GPT implementation. this is true for modern LLMs as well.

total_size_bytes = total_params * 4 # 4 bytes for float32 # we assume that each paramter takes 4 bytes, which is the size of a 32-bit floating point number (float32). This is a common data type used for model parameters in deep learning.
total_size_mb = total_size_bytes / (1024 * 1024) # convert bytes
print(f"Total size of the model in MB:{total_size_mb: .2f} MB")
# Total size of the model in MB: 621.83 MB
# this illustrates the memory requirements for storing the model's parameters, which is important for understanding the hardware requirements for training and inference. and we need relatively high-end GPUs with sufficient VRAM to handle the model, especially for training.
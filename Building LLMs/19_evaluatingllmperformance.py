# with big dataset the training could take even weeks to complete, so we keep the dataset small
import os
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #shorten for the trianing simplicity
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}
tokenizer = tiktoken.get_encoding("gpt2")

file_path="the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# print(text_data[:99])

# print(text_data[-99:])
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

# print("Characters:", total_characters)
# print("Tokens:", total_tokens)
# Characters: 20758
# Tokens: 5765

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension 
    return tokenizer.decode(flat.tolist())

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        #use sliding window to chunk the book into overlapping sequences of max_length 
        for i in range(0, len(token_ids) - max_length, stride):
            # stop early so every chunk has max_length input tokens + 1 extra token for the shifted target
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
            return len(self.input_ids)

    #DataLoader class in PyTorch uses the __getitem__ method to retrieve a single data point from the dataset. This method takes an index idx as input and returns the corresponding input and target tensors from the input_ids and target_ids lists. The DataLoader will call this method for each index in the dataset, allowing it to efficiently load batches of data during training.
    def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        #batch_size is how many cpu process u would like to run parallely, stride is how many previous tokens to skip to create the next, num_workers is number of cpu threads that we can run simultaneously to load the data, drop_last is whether to drop the last batch if it is smaller than the batch size, shuffle is whether to shuffle the data before creating batches
        #Intialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

        #Create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        #Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        #will see the getitem method in dataset
        # batch size is like the number of chunks the model will see at once and then update its parameters
    return dataloader

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
     train_data, 
     batch_size = 2,
     max_length=GPT_CONFIG_124M["context_length"],
     stride=GPT_CONFIG_124M["context_length"],
     drop_last=True,
     shuffle=True,
     num_workers=0
)

val_loader = create_dataloader_v1(
     val_data, 
     batch_size = 2,
     max_length=GPT_CONFIG_124M["context_length"],
     stride=GPT_CONFIG_124M["context_length"],
     drop_last=True,
     shuffle=True,
     num_workers=0
)

#sanity check 
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
     print("Not enough tokens for the training loader.")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
     print("Not enough tokens for the validation loader.")

# print("Train loader:")
# for x, y in train_loader:
#      print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#      print(x.shape, y.shape)

# print(len(train_loader))
# Train loader:
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])

# Validation loader:
# torch.Size([2, 256]) torch.Size([2, 256])
# 10

train_tokens = 0
for input_batch, target_batch in train_loader:
     train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
     val_tokens += input_batch.numel()

# print("Training tokens:", train_tokens)
# print("Validation tokens:", val_tokens)
# print("All tokens:", train_tokens + val_tokens)
# Training tokens: 5120
# Validation tokens: 512
# All tokens: 5632
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

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Moves the data to the same device as the model (CPU or GPU). PyTorch errors if the model and the data are on different devices.
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0. #. makes it float
    if len(data_loader) == 0:
        return float("nan") #loader has no batches (too littel text), return NaN instead of dividing it by 0
    elif num_batches is None:
        num_batches = len(data_loader) #If you didn’t say how many batches to use, use all of them (10 for train, 1 for val).
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # If you did, e.g. num_batches=5, make sure it isn’t more than the loader has. This is useful later for quick evaluation during training.
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() #.item() turns the 1-element tensor into a plain Python number.
        else:
            break 
    return total_loss / num_batches
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")


model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
# Training loss: 10.970044422149659
# Validation loss: 10.984174728393555